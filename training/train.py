import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import cv2

import ray
import ray.train as train
from ray.train import ScalingConfig, Checkpoint, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, PeftModel

from data_pipeline import get_train_ds, get_val_ds


cv2.setNumThreads(0)
torch.set_num_threads(3)
torch.set_num_interop_threads(2)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NCCL_TIMEOUT"] = "7200000"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "20000"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

os.environ["RAY_gcs_server_rpc_server_thread_num"] = "2"
os.environ["RAY_gcs_server_rpc_client_thread_num"] = "2"
os.environ["RAY_num_server_call_thread"] = "2"


def setup_directories(cfg: DictConfig):
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpointing.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def build_datasets(cfg: DictConfig, preprocessor):
    train_ds = get_train_ds(
        cfg.data.json_annotations_train,
        Path(cfg.data.images_dir_train),
        preprocessor,
        cfg.image_tokens_number,
    )
    val_ds = get_val_ds(
        cfg.data.json_annotations_val,
        Path(cfg.data.images_dir_val),
        preprocessor,
        cfg.image_tokens_number,
    )
    return train_ds, val_ds


def set_seeds(seed: int):
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)


def build_accelerator(mixed_precision: str, grad_accum_steps: int, tb_dir: str):
    from accelerate import Accelerator, DistributedDataParallelKwargs  # inside worker
    from torch.utils.tensorboard import SummaryWriter  # inside worker

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=tb_dir,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=int(grad_accum_steps),
    )
    writer = SummaryWriter(log_dir=tb_dir) if accelerator.is_main_process else None
    return accelerator, writer


def build_model_and_optim(config: Dict[str, Any]):
    # Base model
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
    return model, optimizer, scheduler


def get_datasets():
    train_it = train.get_dataset_shard("train")
    val_it = train.get_dataset_shard("val")
    return train_it, val_it


def restore_from_checkpoint(
    accelerator, model, optimizer, scheduler, config: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Tries restore in order:
    1) train.get_checkpoint() from this run
    2) config['start_with_checkpoint'] path

    Supports two formats:
    - adapter-only: ckpt/adapter/* + training_state.pt
    - full-weights: ckpt/model/pytorch_model.bin + training_state.pt (fallback)
    """
    epoch_start, global_step = 0, 0
    maybe_ckpt = train.get_checkpoint()
    if maybe_ckpt is None and config.get("start_with_checkpoint"):
        p = config["start_with_checkpoint"]
        if isinstance(p, str) and p:
            maybe_ckpt = Checkpoint.from_directory(p)

    if maybe_ckpt is None:
        return epoch_start, global_step

    ckpt_dir = Path(maybe_ckpt.to_directory())
    core = accelerator.unwrap_model(model)

    # Prefer adapter-only format
    adapter_dir = ckpt_dir / "adapter"
    state_path = ckpt_dir / "training_state.pt"
    full_weights = ckpt_dir / "model" / "pytorch_model.bin"

    if adapter_dir.exists():
        # Rebuild base + adapter for exact behavior
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            config["model_name"], dtype=torch.bfloat16
        )
        restored = PeftModel.from_pretrained(base, adapter_dir)
        # Replace in-place
        for p_old, p_new in zip(core.parameters(), restored.parameters()):
            p_old.data.copy_(p_new.data)
    elif full_weights.exists():
        # Fallback: load full state into core
        state_dict = torch.load(full_weights, map_location="cpu")
        core.load_state_dict(state_dict)
    else:
        # Nothing to load for model; continue with training_state only if present
        pass

    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(saved.get("optimizer_state_dict", {}))
        scheduler.load_state_dict(saved.get("scheduler_state_dict", {}))
        epoch_start = int(saved.get("epoch", 0))
        global_step = int(saved.get("global_step", 0))

    accelerator.wait_for_everyone()
    return epoch_start, global_step


def to_device(batch: dict, device: torch.device) -> dict:
    import numpy as np

    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, (np.ndarray,)):
            out[k] = torch.from_numpy(v).to(device)
        else:
            out[k] = v
    return out


def run_validation(
    accelerator,
    model,
    val_it,
    batch_size: int,
    prefetch_batches: int,
    max_batches: int,
    global_step: int,
    writer,
) -> float:
    model.eval()
    total_val_loss, val_batches = 0.0, 0
    with torch.no_grad():
        for i, vbatch in enumerate(
            val_it.iter_torch_batches(
                batch_size=batch_size, prefetch_batches=prefetch_batches, drop_last=True
            )
        ):
            if i >= max_batches:
                break
            vbatch = to_device(vbatch, accelerator.device)
            vout = model(**vbatch)
            total_val_loss += float(vout.loss.item())
            val_batches += 1
    val_loss = total_val_loss / max(1, val_batches)
    model.train()
    if writer:
        writer.add_scalar("val/loss", val_loss, global_step)
    return val_loss


def save_step_checkpoint(
    accelerator, model, optimizer, scheduler, epoch: int, global_step: int
):
    # Save HF‑compatible PEFT adapter ONLY + training_state.pt
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # ensure we operate on core; require PeftModel for adapter-only save
        core = accelerator.unwrap_model(model)
        if not isinstance(core, PeftModel):
            raise TypeError(f"Expected PeftModel after unwrap, got: {type(core)}")
        adapter_dir = tmp / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        core.save_pretrained(
            adapter_dir
        )  # saves only adapter weights + adapter_config.json

        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            tmp / "training_state.pt",
        )

        ckpt = Checkpoint.from_directory(tmp)
        train.report(
            {"epoch": epoch, "global_step": global_step, "step_checkpoint": True},
            checkpoint=ckpt,
        )


def train_epoch(
    accelerator,
    model,
    optimizer,
    scheduler,
    train_it,
    val_it,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    writer,
) -> Tuple[int, float, int]:
    epoch_loss, num_batches = 0.0, 0

    for batch in train_it.iter_torch_batches(
        batch_size=config["batch_size"],
        prefetch_batches=config["prefetch_batches"],
        local_shuffle_buffer_size=config["local_shuffle_buffer_size"],
        drop_last=True,
    ):
        with accelerator.accumulate(model):
            batch = to_device(batch, accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            loss_value = float(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            epoch_loss += loss_value
            num_batches += 1

            if writer and (global_step % config["log_every_n_steps"] == 0):
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar(
                    "train/learning_rate", scheduler.get_last_lr()[0], global_step
                )

            if global_step % config["val_every_n_steps"] == 0:
                val_loss = None
                if accelerator.is_main_process:
                    val_loss = run_validation(
                        accelerator,
                        model,
                        val_it,
                        config["batch_size"],
                        config["prefetch_batches"],
                        config["val_batches"],
                        global_step,
                        writer,
                    )
                train.report(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": loss_value,
                        "val_loss": val_loss,
                    }
                )

            if config["save_every_n_steps"] > 0 and (
                global_step % config["save_every_n_steps"] == 0
            ):
                if accelerator.is_main_process:
                    save_step_checkpoint(
                        accelerator, model, optimizer, scheduler, epoch, global_step
                    )

    avg_epoch_loss = epoch_loss / max(1, num_batches)
    if writer and accelerator.is_main_process:
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

    return global_step, avg_epoch_loss, num_batches


def train_loop(config: Dict[str, Any]):

    set_seeds(int(config["seed"]))

    accelerator, writer = build_accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        tb_dir=config["tensorboard_dir"],
        grad_accum_steps=config["grad_accum_steps"],
    )

    model, optimizer, scheduler = build_model_and_optim(config)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.train()

    train_it, val_it = get_datasets()

    epoch_start, global_step = restore_from_checkpoint(
        accelerator, model, optimizer, scheduler, config
    )

    for epoch in range(epoch_start, config["num_epochs"]):
        global_step, avg_epoch_loss, num_batches = train_epoch(
            accelerator,
            model,
            optimizer,
            scheduler,
            train_it,
            val_it,
            config,
            epoch,
            global_step,
            writer,
        )

    if writer and accelerator.is_main_process:
        writer.close()


@hydra.main(config_path="../configs/train", config_name="default", version_base=None)
def main(cfg: DictConfig):

    from ray.data import ExecutionOptions, ExecutionResources

    ray.init(address=cfg.ray.address,
             runtime_env={
                 "excludes": [
                     "deploy/",
                     "assets/",
                     "docs/",
                     ".git/"
                 ]
             }
    )

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
        "steps_per_epoch_hint": int(
            getattr(cfg.training, "steps_per_epoch_hint", 1000)
        ),
        "mixed_precision": cfg.get("mixed_precision", "bf16"),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
        "start_with_checkpoint": cfg.checkpointing.get("start_with_checkpoint", ""),
        "save_every_n_steps": int(cfg.checkpointing.save_every_n_steps),
        "grad_accum_steps": max(
            1, int(getattr(cfg.training, "grad_accum_steps", 1) or 1)
        ),
        "local_shuffle_buffer_size": (
            int(cfg.training.local_shuffle_buffer_size)
            or 10 * int(cfg.training.batch_size)
        ),
    }

    run_name = getattr(cfg.checkpointing, "run_name", "default_run_name")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=loop_config,
        scaling_config=ScalingConfig(
            num_workers=int(cfg.ray.num_workers),
            use_gpu=True,
            resources_per_worker={"CPU": int(cfg.ray.cpus_per_worker), "GPU": 1},
        ),
        run_config=RunConfig(
            storage_path=cfg.checkpointing.checkpoint_dir,
            name=run_name,
            checkpoint_config=CheckpointConfig(
                num_to_keep=int(cfg.checkpointing.num_to_keep)
            ),
        ),
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=ray.train.DataConfig(
            datasets_to_split=["train"]
        ),
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    main()
