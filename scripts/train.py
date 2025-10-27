import os
import torch
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import ray
from tqdm import tqdm
import numpy as np
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model
from hydra.utils import instantiate
from data_pipeline import get_train_ds, get_val_ds

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"


def setup_directories(cfg: DictConfig):
    """Create necessary directories"""
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpointing.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def make_train_dataset(cfg: DictConfig, preprocessor, world_size, rank):
    """Create training dataset - placeholder for your implementation"""
    # Your dataset creation logic here
    # Should return Ray Dataset
    train_ds = get_train_ds(
        cfg.data.json_annotations_train,
        Path(cfg.data.images_dir_train),
        preprocessor,
        world_size,
        rank,
    )
    return train_ds


def make_val_dataset(cfg: DictConfig, preprocessor):
    """Create validation dataset - placeholder for your implementation"""
    # Your validation dataset creation logic here
    # Should return Ray Dataset
    val_ds = get_val_ds(
        cfg.data.json_annotations_val, Path(cfg.data.images_dir_val), preprocessor
    )
    return val_ds


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Convert numpy arrays to torch tensors and move to device"""
    return {
        k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
        for k, v in batch.items()
    }


def validate(model, val_dataset, accelerator: Accelerator, cfg: DictConfig):
    """Run validation only on rank 0"""
    if accelerator.is_main_process:
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            val_iterator = val_dataset.iter_batches(
                batch_size=cfg.training.batch_size,
                prefetch_batches=cfg.training.prefetch_batches,
            )

            for batch in val_iterator:
                if num_batches >= cfg.training.val_batches:
                    break

                batch = batch_to_device(batch, accelerator.device)

                # Forward pass
                outputs = model(**batch)

                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        model.train()
        return avg_loss

    return None


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    cfg: DictConfig,
    accelerator: Accelerator,
):
    """Save checkpoint"""
    if accelerator.is_main_process:
        checkpoint_path = (
            Path(cfg.checkpointing.checkpoint_dir)
            / f"checkpoint_epoch_{epoch}_step_{global_step}"
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model (unwrap if wrapped by accelerator)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_path)

        # Save optimizer and scheduler states
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            checkpoint_path / "training_state.pt",
        )

        print(f"Checkpoint saved to {checkpoint_path}")


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from ray.data import ExecutionOptions, ExecutionResources

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options = ExecutionOptions(
        resource_limits=ExecutionResources(
            cpu=8, gpu=1, object_store_memory=10e9  # 5GB
        ),
        locality_with_output=True,
        preserve_order=False,
    )

    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_MAX_THREADS": "1",
            }
        },
    )

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Setup directories
    setup_directories(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=cfg.logging.tensorboard_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # Setup TensorBoard writer (only on main process)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=cfg.logging.tensorboard_dir)

    # Load model and tokenizer
    # Replace with your model loading logic

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model, dtype=torch.bfloat16
    )
    preprocessor = PaliGemmaProcessor.from_pretrained(cfg.model, use_fast=True)

    # model = torch.compile(model)  # , dynamic=True

    # Apply LoRA
    lora_config = instantiate(cfg.lora)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Instantiate optimizer with model parameters
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Calculate total training steps
    # Note: This is approximate, update based on your dataset size
    total_steps = cfg.training.num_epochs * 1000  # Placeholder

    # Instantiate scheduler with updated num_training_steps
    scheduler = instantiate(
        cfg.scheduler, optimizer=optimizer, num_training_steps=total_steps
    )

    # Prepare model, optimizer, scheduler with accelerator
    # Note: NOT preparing the dataset as per requirements
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Training loop
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        accelerator.print(f"\n{'='*50}")
        accelerator.print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        accelerator.print(f"{'='*50}")

        # Create fresh dataset for this epoch
        train_dataset = make_train_dataset(
            cfg, preprocessor, accelerator.num_processes, accelerator.process_index
        )

        # Get data iterator
        train_iterator = train_dataset.iter_batches(
            batch_size=cfg.training.batch_size,
            prefetch_batches=cfg.training.prefetch_batches,
            drop_last=True,
        )

        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_iterator,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Convert numpy to torch and move to device manually
                batch = batch_to_device(batch, accelerator.device)

                # Forward pass
                outputs = model(**batch)

                loss = outputs.loss

                # Backward pass
                accelerator.backward(loss)

                # # Gradient clipping
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(
                #         model.parameters(), cfg.training.max_grad_norm
                #     )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                # Log to TensorBoard
                if (
                    global_step % cfg.logging.log_every_n_steps == 0
                    and accelerator.is_main_process
                ):
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/learning_rate", scheduler.get_last_lr()[0], global_step
                    )

                # Validation
                if global_step % cfg.training.val_every_n_steps == 0:
                    val_dataset = make_val_dataset(cfg, preprocessor)
                    val_loss = validate(model, val_dataset, accelerator, cfg)

                    if accelerator.is_main_process and val_loss is not None:
                        writer.add_scalar("val/loss", val_loss, global_step)
                        accelerator.print(f"\nValidation Loss: {val_loss:.4f}")

        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        accelerator.print(f"\nEpoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

        if accelerator.is_main_process:
            writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % cfg.checkpointing.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step, cfg, accelerator
            )

    # Final validation
    accelerator.print("\nRunning final validation...")
    val_dataset = make_val_dataset(cfg)
    final_val_loss = validate(model, val_dataset, accelerator, cfg)

    if accelerator.is_main_process and final_val_loss is not None:
        writer.add_scalar("val/final_loss", final_val_loss, global_step)
        accelerator.print(f"Final Validation Loss: {final_val_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        cfg.training.num_epochs,
        global_step,
        cfg,
        accelerator,
    )

    if accelerator.is_main_process:
        writer.close()

    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()
