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


from ray.train import ScalingConfig, Checkpoint
from ray.air.config import RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from data_pipeline import get_train_ds, get_val_ds

# Limit threading
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

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

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

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config["model_name"], dtype=torch.bfloat16
    )
    lora_cfg = OmegaConf.create(config["lora"])
    model = get_peft_model(model, instantiate(lora_cfg))
    model.print_trainable_parameters()

    optimizer = instantiate(
        OmegaConf.create(config["optimizer"]), params=model.parameters()
    )
    total_steps = config["num_epochs"] * int(config["steps_per_epoch_hint"])
    scheduler = instantiate(
        OmegaConf.create(config["scheduler"]),
        optimizer=optimizer,
        num_training_steps=total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    train_it = train.get_dataset_shard("train")
    val_it = train.get_dataset_shard("val")

    # Checkpoint recovering: first attempting from the current run,
    # then path from start_with_checkpoint if available
    epoch_start = 0
    global_step = 0
    maybe_ckpt = train.get_checkpoint()
    if maybe_ckpt is None and config.get("start_with_checkpoint"):
        if isinstance(config["start_with_checkpoint"], str) and config["start_with_checkpoint"]:
            maybe_ckpt = Checkpoint.from_directory(config["start_with_checkpoint"])

    if maybe_ckpt is not None:
        ckpt_dir = maybe_ckpt.to_directory()
        unwrapped = accelerator.unwrap_model(model)

        weights_path = Path(ckpt_dir) / "model" / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            unwrapped.load_state_dict(state_dict)

        state_path = Path(ckpt_dir) / "training_state.pt"
        if state_path.exists():
            saved = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(saved.get("optimizer_state_dict", {}))
            scheduler.load_state_dict(saved.get("scheduler_state_dict", {}))
            epoch_start = int(saved.get("epoch", 0))
            global_step = int(saved.get("global_step", 0))

        accelerator.wait_for_everyone()

    model.train()

    def to_device(batch: dict, device: torch.device) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, (np.ndarray,)):
                out[k] = torch.from_numpy(v).to(device)
            else:
                out[k] = v
        return out

    for epoch in range(epoch_start, config["num_epochs"]):
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
                        val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
                        model.train()
                        writer.add_scalar("val/loss", val_loss, global_step)

                    # reporting validation
                    train.report(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": loss.item(),
                            "val_loss": val_loss,
                        }
                    )

                # reporting checkpoint
                if accelerator.is_main_process and config["save_every_n_steps"] > 0:
                    if global_step % config["save_every_n_steps"] == 0:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            unwrapped = accelerator.unwrap_model(model)
                            model_dir = Path(tmpdir) / "model"
                            model_dir.mkdir(parents=True, exist_ok=True)

                            unwrapped.save_pretrained(model_dir)

                            torch.save(
                                {
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                },
                                Path(tmpdir) / "training_state.pt",
                            )

                            ckpt = Checkpoint.from_directory(tmpdir)
                            train.report(
                                {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "step_checkpoint": True,
                                },
                                checkpoint=ckpt,
                            )

        # epoch log
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if writer and accelerator.is_main_process:
            writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

    if writer and accelerator.is_main_process:
        writer.close()


@hydra.main(config_path="../configs", config_name="train_ray", version_base=None)
def main(cfg: DictConfig):

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

    ray.init(address=cfg.ray.address)

    setup_directories(cfg)

    preprocessor = PaliGemmaProcessor.from_pretrained(cfg.model, use_fast=True)
    train_ds, val_ds = build_datasets(cfg, preprocessor)

    val_ds = val_ds.materialize()

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
        "steps_per_epoch_hint": int(getattr(cfg.training, "steps_per_epoch_hint", 1000)),
        "mixed_precision": cfg.get("mixed_precision", "bf16"),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
        "start_with_checkpoint": cfg.checkpointing.get("start_with_checkpoint", ""),
        "save_every_n_steps": int(cfg.checkpointing.save_every_n_steps),
    }

    run_name = getattr(cfg.checkpointing, "run_name", "default_run_name")

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
            name=run_name,
            checkpoint_config=CheckpointConfig(
                num_to_keep=int(cfg.checkpointing.num_to_keep)
            ),
        ),
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=train.DataConfig(datasets_to_split=["train"]),
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    main()
