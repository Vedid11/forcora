"""Dataset and DataLoader utilities for FORCORA DistilBERT."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

from src.config import TrainingConfig


class PrivacyReviewDataset(Dataset):
    """PyTorch Dataset for privacy policy reviews.
    
    Reads the raw 'Review' text and integer 'Label' column.
    Tokenizes text on-the-fly or in pre-indexed format with truncation,
    deferring padding to the DataCollator for dynamic batch padding.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        text_column: str = "Review",
        label_column: str = "Label",
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # Load only required columns
        df = pd.read_csv(self.csv_path, usecols=[self.text_column, self.label_column])
        
        # Ensure clean strings and valid labels
        self.texts = df[self.text_column].astype(str).tolist()
        self.labels = df[self.label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize with truncation, without static padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


def get_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with dynamic batch padding.
    
    NOTE: The test split is deliberately omitted from this function to guarantee
    zero contamination during training, validation, and hyperparameter tuning.
    """
    train_dataset = PrivacyReviewDataset(
        csv_path=config.train_path,
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
        text_column=config.text_column,
        label_column=config.label_column,
    )

    val_dataset = PrivacyReviewDataset(
        csv_path=config.val_path,
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
        text_column=config.text_column,
        label_column=config.label_column,
    )

    # Dynamic batch padding collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    is_cuda = config.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,  # Safe cross-platform default on Windows
        pin_memory=is_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=is_cuda,
    )

    return train_loader, val_loader


def get_test_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
) -> DataLoader:
    """Create test DataLoader for final offline post-training evaluation ONLY.
    
    WARNING: Do NOT call or pass this dataloader during model training,
    checkpoint selection, or hyperparameter optimization.
    """
    test_dataset = PrivacyReviewDataset(
        csv_path=config.test_path,
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
        text_column=config.text_column,
        label_column=config.label_column,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    return DataLoader(
        test_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=(config.device == "cuda"),
    )
