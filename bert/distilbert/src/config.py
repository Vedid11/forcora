"""Central configuration and hyperparameter management for FORCORA DistilBERT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import torch


@dataclass
class TrainingConfig:
    # ---------------------------------------------------------
    # Path Configuration (Anchored to Project Root)
    # ---------------------------------------------------------
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    data_dir: Path = field(init=False)
    train_path: Path = field(init=False)
    val_path: Path = field(init=False)
    test_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # ---------------------------------------------------------
    # Model & Architecture Configuration
    # ---------------------------------------------------------
    model_checkpoint: str = "distilbert-base-uncased"
    num_classes: int = 3
    class_names: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Feature Request",
            1: "Bug Report",
            2: "Privacy-Irrelevant"
        }
    )

    # ---------------------------------------------------------
    # Data & Tokenization
    # ---------------------------------------------------------
    text_column: str = "Review"
    label_column: str = "Label"
    max_sequence_length: int = 256
    use_dynamic_padding: bool = True

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2  # Effective batch size = 8 * 2 = 16
    effective_batch_size: int = 16

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    random_seed: int = 42

    # ---------------------------------------------------------
    # Precision & Hardware
    # ---------------------------------------------------------
    fp16: bool = True  # Automatically checked against CUDA availability
    device: str = field(init=False)

    # ---------------------------------------------------------
    # Checkpoint & Evaluation Strategy
    # ---------------------------------------------------------
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    save_total_limit: int = 2

    def __post_init__(self):
        # Resolve data paths
        self.data_dir = self.project_root / "DATA" / "RAW" / "privacy_reviews"

        # Check for both filename variants
        train_file_primary = self.data_dir / "original_12756 (1).csv"
        train_file_alt = self.data_dir / "original_12756.csv"
        self.train_path = train_file_primary if train_file_primary.exists() else train_file_alt

        self.val_path = self.data_dir / "val_1595.csv"
        self.test_path = self.data_dir / "test_1594.csv"

        # Output and logging directories
        self.output_dir = self.project_root / "outputs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        # Hardware & precision setup
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        self.fp16 = self.fp16 and cuda_available
        self.effective_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        return {
            "model_checkpoint": self.model_checkpoint,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "max_sequence_length": self.max_sequence_length,
            "use_dynamic_padding": self.use_dynamic_padding,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "random_seed": self.random_seed,
            "fp16": self.fp16,
            "device": self.device,
            "metric_for_best_model": self.metric_for_best_model,
            "train_path": str(self.train_path),
            "val_path": str(self.val_path),
            "test_path": str(self.test_path),
            "output_dir": str(self.output_dir),
        }
