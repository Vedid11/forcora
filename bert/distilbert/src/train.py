"""DistilBERT training engine for FORCORA with FP16, Gradient Accumulation, and Checkpointing."""

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.config import TrainingConfig
from src.dataset import get_dataloaders
from src.metrics import compute_classification_metrics

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("FORCORA.Train")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed fixed to {seed}")


class DistilBertTrainer:
    """Modular training engine for DistilBERT sequence classification."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        set_seed(self.config.random_seed)

        self.device = torch.device(self.config.device)
        logger.info(f"Target compute device: {self.device}")

        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint,
            use_fast=True,
        )

        # 2. Build Model
        logger.info(
            f"Initializing DistilBertForSequenceClassification ({self.config.num_classes} classes)"
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_checkpoint,
            num_labels=self.config.num_classes,
            id2label=self.config.class_names,
            label2id={v: k for k, v in self.config.class_names.items()},
        )
        self.model.to(self.device)

        # 3. Setup DataLoaders (Train and Validation only; Test is excluded)
        logger.info("Initializing train and validation DataLoaders with dynamic padding...")
        self.train_loader, self.val_loader = get_dataloaders(
            tokenizer=self.tokenizer,
            config=self.config,
        )
        logger.info(
            f"DataLoaders ready: Train batches={len(self.train_loader)}, Val batches={len(self.val_loader)}"
        )

        # 4. Optimizer with weight decay exclusion for bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8,
        )

        # 5. Learning Rate Scheduler with exact math.ceil update calculation
        self.steps_per_epoch = math.ceil(
            len(self.train_loader) / self.config.gradient_accumulation_steps
        )
        total_update_steps = self.steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_update_steps * self.config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
        logger.info(
            f"Scheduler configured: steps_per_epoch={self.steps_per_epoch}, total_steps={total_update_steps}, warmup_steps={warmup_steps}"
        )

        # 6. Mixed Precision Scaler
        self.use_fp16 = self.config.fp16 and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)
        logger.info(f"FP16 Mixed Precision enabled: {self.use_fp16}")

        # Tracking state
        self.start_epoch = 1
        self.best_val_macro_f1 = -1.0
        self.training_history: List[Dict[str, Any]] = []

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        val_metrics: Dict[str, Any],
        is_best: bool = False,
    ):
        """Save complete checkpoint: model weights, tokenizer, optimizer, scheduler, and full RNG states."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save Hugging Face model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Capture complete RNG states for bit-exact reproducibility
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

        # Save PyTorch training states
        state = {
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_macro_f1": self.best_val_macro_f1,
            "val_metrics": val_metrics,
            "training_history": self.training_history,
            "config": self.config.to_dict(),
            "rng_state": rng_state,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

        with open(checkpoint_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        tag = "BEST MODEL" if is_best else f"epoch {epoch}"
        logger.info(f"Saved {tag} checkpoint to: {checkpoint_dir}")

    def resume_from_checkpoint(self, checkpoint_path: Path):
        """Resume training state from a previously saved checkpoint with backward compatibility."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        logger.info(f"Resuming pipeline state from: {checkpoint_path}")

        # Reload model weights
        self.model = DistilBertForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=self.config.num_classes,
            id2label=self.config.class_names,
            label2id={v: k for k, v in self.config.class_names.items()},
        )
        self.model.to(self.device)

        # Reload optimizer, scheduler, scaler, and RNG states
        state_file = checkpoint_path / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            self.start_epoch = state["epoch"] + 1
            self.best_val_macro_f1 = state.get("best_val_macro_f1", -1.0)
            self.training_history = state.get("training_history", [])

            # Restore RNG states with backward compatibility
            if "rng_state" in state and state["rng_state"] is not None:
                rng_state = state["rng_state"]
                if "python" in rng_state and rng_state["python"] is not None:
                    random.setstate(rng_state["python"])
                if "numpy" in rng_state and rng_state["numpy"] is not None:
                    np.random.set_state(rng_state["numpy"])
                if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
                    cpu_state = rng_state["torch_cpu"]
                    if isinstance(cpu_state, torch.Tensor):
                        cpu_state = cpu_state.cpu()
                    torch.set_rng_state(cpu_state)
                if (
                    torch.cuda.is_available()
                    and "torch_cuda" in rng_state
                    and rng_state["torch_cuda"] is not None
                ):
                    try:
                        cuda_states = [
                            x.cpu() if isinstance(x, torch.Tensor) else x
                            for x in rng_state["torch_cuda"]
                        ]
                        torch.cuda.set_rng_state_all(cuda_states)
                    except Exception as e:
                        logger.warning(f"Could not restore CUDA RNG state: {e}")
                logger.info("Restored Python, NumPy, torch CPU, and CUDA RNG states from checkpoint.")
            else:
                logger.info("No RNG state found in checkpoint (backward compatibility maintained).")

            logger.info(
                f"Resumed from epoch {state['epoch']} (next: {self.start_epoch}). Best Val Macro-F1: {self.best_val_macro_f1:.4f}"
            )
        else:
            logger.warning("training_state.pt not found; resuming only model weights.")

    def train_epoch(self, epoch: int) -> float:
        """Execute one training epoch with mathematically exact gradient accumulation and FP16 AMP."""
        self.model.train()
        total_loss = 0.0
        window_loss = 0.0
        window_steps = 0
        optimizer_step_count = 0
        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        accum_steps = self.config.gradient_accumulation_steps

        epoch_start_time = time.time()

        for step, batch in enumerate(self.train_loader, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Determine accumulation divisor for current batch:
            # If this is the final batch and it forms an incomplete accumulation window,
            # normalize by the remaining batch count (1), ensuring the batch is not underweighted.
            is_final_batch = (step == num_batches)
            is_orphan_batch = is_final_batch and (num_batches % accum_steps != 0)
            current_accum_steps = 1 if is_orphan_batch else accum_steps

            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=self.use_fp16,
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                raw_loss = outputs.loss
                # Scale loss by the number of batches in this accumulation window
                loss = raw_loss / current_accum_steps

            self.scaler.scale(loss).backward()

            total_loss += raw_loss.item()
            window_loss += raw_loss.item()
            window_steps += 1

            # Update weights on accumulation boundary or epoch end
            if step % accum_steps == 0 or is_final_batch:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                optimizer_step_count += 1

                if optimizer_step_count % 100 == 0 or is_final_batch:
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_step_loss = window_loss / max(window_steps, 1)
                    logger.info(
                        f"Epoch {epoch} | Step {step}/{num_batches} (Opt Step {optimizer_step_count}/{self.steps_per_epoch}) | Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}"
                    )
                    window_loss = 0.0
                    window_steps = 0

        avg_epoch_loss = total_loss / num_batches
        duration = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch} Train Completed in {duration:.1f}s | Average Train Loss: {avg_epoch_loss:.4f} | Total Optimizer Steps: {optimizer_step_count}"
        )
        return avg_epoch_loss

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on validation dataset using class-aware metrics."""
        self.model.eval()
        total_eval_loss = 0.0
        all_preds = []
        all_labels = []

        eval_start_time = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=torch.float16,
                    enabled=self.use_fp16,
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                total_eval_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = total_eval_loss / len(self.val_loader)
        eval_duration = time.time() - eval_start_time

        # Calculate metrics
        metrics = compute_classification_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            num_classes=self.config.num_classes,
            class_names=self.config.class_names,
        )
        metrics["val_loss"] = round(avg_val_loss, 4)
        metrics["eval_duration_sec"] = round(eval_duration, 2)

        return metrics

    def run(self, resume_from: Optional[Path] = None):
        """Execute the full training pipeline across configured epochs."""
        if resume_from:
            self.resume_from_checkpoint(resume_from)

        # Save initial configuration
        config_path = self.config.output_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        pipeline_start_time = time.time()
        logger.info(
            f"Starting DistilBERT Training | Epochs: {self.start_epoch} to {self.config.num_epochs}"
        )

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            logger.info("=" * 65)
            logger.info(f"STARTING EPOCH {epoch}/{self.config.num_epochs}")
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate on Validation Set
            logger.info(f"Running Validation Evaluation for Epoch {epoch}...")
            val_metrics = self.evaluate()
            epoch_duration = time.time() - epoch_start

            val_macro_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["val_loss"]

            logger.info(
                f"\nEpoch {epoch} Summary:\n"
                f"  Train Loss:       {train_loss:.4f}\n"
                f"  Val Loss:         {val_loss:.4f}\n"
                f"  Val Accuracy:     {val_acc:.4f}\n"
                f"  Val Macro-F1:     {val_macro_f1:.4f}\n"
                f"  Epoch Time:       {epoch_duration:.1f}s\n"
                f"\nValidation Classification Report:\n{val_metrics['summary_table']}"
            )

            # Check if this is the best model so far
            is_best = val_macro_f1 > self.best_val_macro_f1
            if is_best:
                self.best_val_macro_f1 = val_macro_f1
                logger.info(
                    f"★ NEW BEST MODEL! Validation Macro-F1: {self.best_val_macro_f1:.4f}"
                )

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "epoch_duration_sec": round(epoch_duration, 2),
                "is_best": is_best,
            }
            self.training_history.append(epoch_record)

            # Save epoch checkpoint
            epoch_ckpt_dir = (
                self.config.checkpoint_dir / f"checkpoint-epoch-{epoch}"
            )
            self.save_checkpoint(
                epoch_ckpt_dir,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=False,
            )

            # Save best checkpoint
            if is_best:
                best_ckpt_dir = self.config.checkpoint_dir / "best_model"
                self.save_checkpoint(
                    best_ckpt_dir,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    is_best=True,
                )

            # Update history file
            with open(
                self.config.logs_dir / "training_history.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.training_history, f, indent=2)

        total_time = time.time() - pipeline_start_time
        logger.info("=" * 65)
        logger.info(
            f"TRAINING COMPLETED SUCCESSFULLY in {total_time / 60:.2f} minutes."
        )
        logger.info(f"Best Validation Macro-F1 Achieved: {self.best_val_macro_f1:.4f}")
        logger.info(
            f"Best checkpoint saved at: {self.config.checkpoint_dir / 'best_model'}"
        )
