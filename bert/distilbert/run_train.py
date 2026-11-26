"""FORCORA DistilBERT Training CLI Entrypoint."""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to sys.path so 'src' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.train import DistilBertTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT on FORCORA privacy policy reviews."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size (default: 8)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, effective batch size = 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="AdamW learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length with dynamic padding (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16 mixed precision",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate environment, data loading, and model initialization without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_sequence_length=args.max_length,
        random_seed=args.seed,
        fp16=not args.no_fp16,
    )

    print("=" * 65)
    print("FORCORA DISTILBERT TRAINING PIPELINE INITIALIZATION")
    print("=" * 65)
    print(f"Project root:               {config.project_root}")
    print(f"Train dataset:              {config.train_path} (exists: {config.train_path.exists()})")
    print(f"Validation dataset:         {config.val_path} (exists: {config.val_path.exists()})")
    print(f"Test dataset (ISOLATED):    {config.test_path} (exists: {config.test_path.exists()})")
    print(f"Output directory:           {config.output_dir}")
    print(f"Device:                     {config.device}")
    print(f"FP16 enabled:               {config.fp16}")
    print(f"Sequence length ceiling:    {config.max_sequence_length} (Dynamic Padding enabled)")
    print(f"Batch size per device:      {config.per_device_train_batch_size}")
    print(f"Gradient accumulation:      {config.gradient_accumulation_steps}")
    print(f"Effective batch size:       {config.effective_batch_size}")
    print(f"Learning rate:              {config.learning_rate}")
    print(f"Epochs:                     {config.num_epochs}")
    print(f"Random seed:                {config.random_seed}")
    print("=" * 65)

    trainer = DistilBertTrainer(config=config)

    if args.dry_run:
        print("\n[DRY RUN MODE] Verifying pipeline components without executing training:")
        print("  - Checking one sample batch from train DataLoader...")
        sample_batch = next(iter(trainer.train_loader))
        print(f"  - Sample input_ids shape:      {sample_batch['input_ids'].shape}")
        print(f"  - Sample attention_mask shape: {sample_batch['attention_mask'].shape}")
        print(f"  - Sample labels shape:         {sample_batch['labels'].shape}")
        print("  - Performing test forward pass (no backward pass)...")
        with torch.no_grad():
            inp = sample_batch['input_ids'].to(trainer.device)
            att = sample_batch['attention_mask'].to(trainer.device)
            lbl = sample_batch['labels'].to(trainer.device)
            out = trainer.model(input_ids=inp, attention_mask=att, labels=lbl)
            print(f"  - Test loss calculated:        {out.loss.item():.4f}")
            print(f"  - Test logits shape:           {out.logits.shape}")
        print("\n[DRY RUN PASS] Pipeline verified successfully. Ready for training.")
        return

    # Execute full training
    resume_path = Path(args.resume_from) if args.resume_from else None
    trainer.run(resume_from=resume_path)


if __name__ == "__main__":
    main()
