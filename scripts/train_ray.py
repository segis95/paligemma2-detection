import os
import torch
import hydra
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import ray
import ray.train as train
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model

from ray.air import session
from ray.train import ScalingConfig
from ray.air.config import RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer  # <-- замена AccelerateTrainer

from data_pipeline import get_train_ds, get_val_ds

# При необходимости можно включить ограничения по потокам
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["PYTORCH_NUM_THREADS"] = "1"


def setup_directories(cfg: DictConfig):
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpointing.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def build_datasets(cfg: DictConfig, preprocessor):
    # Возвращаем Ray Dataset'ы; шардирование сделает Ray на воркерах
    train_ds = get_train_ds(
        cfg.data.json_annotations_train,
        Path(cfg.data.images_dir_train),
        preprocessor,
    )
    val_ds = get_val_ds(
        cfg.data.json_annotations_val,
        Path(cfg.data.images_dir_val),
        preprocessor,
    )
    return train_ds, val_ds


def train_loop(config: Dict[str, Any]):
    import numpy as np
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from torch.utils.tensorboard import SummaryWriter

    # Сидирование
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Accelerator с DDP kwargs; DDP окружение поднимет TorchTrainer
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        log_with="tensorboard",
        project_dir=config["tensorboard_dir"],
        kwargs_handlers=[ddp_kwargs],
    )

    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=config["tensorboard_dir"])

    # Модель и процессор (dtype bf16 как было)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config["model_name"], dtype=torch.bfloat16
    )
    # LoRA
    lora_cfg = OmegaConf.create(config["lora"])
    model = get_peft_model(model, instantiate(lora_cfg))
    model.print_trainable_parameters()

    # Оптимизатор и шедулер
    optimizer = instantiate(
        OmegaConf.create(config["optimizer"]), params=model.parameters()
    )
    total_steps = config["num_epochs"] * int(config["steps_per_epoch_hint"])
    scheduler = instantiate(
        OmegaConf.create(config["scheduler"]),
        optimizer=optimizer,
        num_training_steps=total_steps,
    )

    # Подготовка через Accelerator
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Получаем шарды Ray Data на воркере
    train_it = session.get_dataset_shard("train")
    val_it = session.get_dataset_shard("val")

    global_step = 0
    model.train()

    def to_device(batch: dict, device: torch.device) -> dict:
        # Универсально переносим и torch.Tensor, и numpy.ndarray
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, (np.ndarray,)):
                out[k] = torch.from_numpy(v).to(device)
            else:
                out[k] = v
        return out

    # Обучение по эпохам
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_it.iter_torch_batches(
                batch_size=config["batch_size"],
                prefetch_batches=config["prefetch_batches"],
                drop_last=True,
        ):
            with accelerator.accumulate(model):
                batch = to_device(batch, accelerator.device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pixel_values=batch["pixel_values"],
                )

                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1

                if writer and (global_step % config["log_every_n_steps"] == 0):
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/learning_rate", scheduler.get_last_lr()[0], global_step
                    )

                # Периодическая валидация
                if global_step % config["val_every_n_steps"] == 0:
                    val_loss = None
                    if accelerator.is_main_process:
                        model.eval()
                        total_val_loss = 0.0
                        val_batches = 0
                        with torch.no_grad():
                            for i, vbatch in enumerate(
                                    val_it.iter_torch_batches(
                                        batch_size=config["batch_size"],
                                        prefetch_batches=config["prefetch_batches"],
                                        drop_last=True,
                                    )
                            ):
                                if i >= config["val_batches"]:
                                    break
                                vbatch = to_device(vbatch, accelerator.device)
                                vout = model(
                                    input_ids=vbatch["input_ids"],
                                    attention_mask=vbatch["attention_mask"],
                                    labels=vbatch["labels"],
                                    pixel_values=vbatch["pixel_values"],
                                )
                                total_val_loss += vout.loss.item()
                                val_batches += 1
                        val_loss = (
                            total_val_loss / val_batches if val_batches > 0 else 0.0
                        )
                        model.train()
                        writer.add_scalar("val/loss", val_loss, global_step)

                    # Репортим метрики (и при необходимости чекпоинт ниже)
                    session.report(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": loss.item(),
                            "val_loss": val_loss,
                        }
                    )

        # Лог по эпохе
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if writer and accelerator.is_main_process:
            writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

        # Чекпоинт по эпохам (rank 0)
        if accelerator.is_main_process and (
                (epoch + 1) % config["save_every_n_epochs"] == 0
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                unwrapped = accelerator.unwrap_model(model)
                model_dir = Path(tmpdir) / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                unwrapped.save_pretrained(model_dir)

                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    Path(tmpdir) / "training_state.pt",
                )

                ckpt = train.Checkpoint.from_directory(tmpdir)
                session.report(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "avg_epoch_loss": avg_epoch_loss,
                    },
                    checkpoint=ckpt,
                )

    if writer and accelerator.is_main_process:
        writer.close()


@hydra.main(config_path="../configs", config_name="train_ray", version_base=None)
def main(cfg: DictConfig):
    # Настройка Ray Data контекста (опционально)
    from ray.data import ExecutionOptions, ExecutionResources

    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options = ExecutionOptions(
        resource_limits=ExecutionResources(
            cpu=cfg.ray.driver_cpu,
            gpu=cfg.ray.driver_gpu,
            object_store_memory=cfg.ray.object_store_mem,
        ),
        locality_with_output=True,
        preserve_order=False,
    )

    # Коннект к кластеру Ray
    ray.init(address=cfg.ray.address)

    # Директории и конфиг
    setup_directories(cfg)

    # Процессор и датасеты на драйвере
    preprocessor = PaliGemmaProcessor.from_pretrained(cfg.model, use_fast=True)
    train_ds, val_ds = build_datasets(cfg, preprocessor)

    # Параметры для train_loop
    loop_config = {
        "seed": int(cfg.seed),
        "tensorboard_dir": cfg.logging.tensorboard_dir,
        "model_name": cfg.model,
        "num_epochs": int(cfg.training.num_epochs),
        "batch_size": int(cfg.training.batch_size),
        "prefetch_batches": int(cfg.training.prefetch_batches),
        "log_every_n_steps": int(cfg.logging.log_every_n_steps),
        "val_every_n_steps": int(cfg.training.val_every_n_steps),
        "val_batches": int(cfg.training.val_batches),
        "save_every_n_epochs": int(cfg.checkpointing.save_every_n_epochs),
        "steps_per_epoch_hint": int(getattr(cfg.training, "steps_per_epoch_hint", 1000)),
        "mixed_precision": cfg.get("mixed_precision", "bf16"),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
    }

    # TorchTrainer вместо AccelerateTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=loop_config,
        scaling_config=ScalingConfig(
            num_workers=int(cfg.ray.num_workers),
            use_gpu=True,
            resources_per_worker={"CPU": int(cfg.ray.cpus_per_worker)},
        ),
        run_config=RunConfig(
            storage_path=cfg.checkpointing.checkpoint_dir,
            checkpoint_config=CheckpointConfig(
                num_to_keep=int(cfg.checkpointing.num_to_keep)
            ),
        ),
        datasets={"train": train_ds, "val": val_ds},
        # Сплитим только train (валидацию не сплитим)
        dataset_config=train.DataConfig(datasets_to_split=["train"]),
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    main()
