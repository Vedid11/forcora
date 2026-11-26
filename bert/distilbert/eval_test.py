"""FORCORA — Final DistilBERT Test Set Evaluation Script.

Evaluates the selected best_model checkpoint on test_1594.csv exactly once.
Computes test metrics, per-class performance, confusion matrix, and latency.
Saves all artifacts into outputs/test_evaluation/.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification

from src.metrics import compute_classification_metrics

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("FORCORA.TestEval")

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model"
TEST_DATA_PATH = PROJECT_ROOT / "DATA" / "RAW" / "privacy_reviews" / "test_1594.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_evaluation"

CLASS_NAMES = {
    0: "Feature Request",
    1: "Bug Report",
    2: "Privacy-Irrelevant",
}


class TestDataset(Dataset):
    """PyTorch Dataset reading raw Review text and Label for final test evaluation."""

    def __init__(self, csv_path: Path, tokenizer, max_length: int = 256):
        df = pd.read_csv(csv_path, usecols=["Review", "Label"])
        self.texts = df["Review"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Generate and save publication-quality confusion matrix image."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="DistilBERT Final Test Set Confusion Matrix (N=1,594)",
        ylabel="Ground Truth Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
                weight="bold",
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to: {save_path}")


def main():
    logger.info("=" * 65)
    logger.info("FORCORA — FINAL DISTILBERT TEST SET EVALUATION")
    logger.info("=" * 65)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at: {TEST_DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Device Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    logger.info(f"Compute Device: {device} ({gpu_name})")

    # 2. Load Tokenizer & Model
    logger.info(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, use_fast=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH,
        num_labels=3,
        id2label=CLASS_NAMES,
        label2id={v: k for k, v in CLASS_NAMES.items()},
    )
    model.to(device)
    model.eval()

    # 3. Load Test Data
    logger.info(f"Loading test dataset from: {TEST_DATA_PATH}")
    test_dataset = TestDataset(TEST_DATA_PATH, tokenizer=tokenizer, max_length=256)
    total_test_samples = len(test_dataset)
    logger.info(f"Total test samples to evaluate: {total_test_samples}")
    assert total_test_samples == 1594, f"Expected 1,594 samples, got {total_test_samples}"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=use_cuda,
    )

    # 4. Execute Inference
    logger.info("Starting inference on test set...")
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    all_preds = []
    all_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(
                device_type="cuda" if use_cuda else "cpu",
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total_duration = time.perf_counter() - start_time
    avg_latency_ms = (total_duration / total_test_samples) * 1000.0

    peak_vram_mb = 0.0
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    logger.info(f"Inference completed in {total_duration:.2f}s ({avg_latency_ms:.2f} ms/sample)")
    if use_cuda:
        logger.info(f"Peak VRAM during inference: {peak_vram_mb:.1f} MiB")

    # 5. Compute Metrics
    metrics = compute_classification_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        num_classes=3,
        class_names=CLASS_NAMES,
    )

    cm = np.array(metrics["confusion_matrix"])

    # 6. Prepare Full Metadata Output
    full_metrics = {
        "dataset": "test_1594.csv",
        "num_samples": total_test_samples,
        "checkpoint_evaluated": str(CHECKPOINT_PATH),
        "device": str(device),
        "gpu_name": gpu_name,
        "inference_duration_seconds": round(total_duration, 4),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 2),
        "peak_vram_allocated_mb": round(peak_vram_mb, 1),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    # 7. Save Artifacts
    # Artifact 1: test_metrics.json
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved: {metrics_path}")

    # Artifact 2: test_classification_report.json
    report_path = OUTPUT_DIR / "test_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "per_class": metrics["per_class"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved: {report_path}")

    # Artifact 3: test_confusion_matrix.png
    cm_plot_path = OUTPUT_DIR / "test_confusion_matrix.png"
    class_labels = [CLASS_NAMES[i] for i in range(3)]
    plot_confusion_matrix(cm, class_labels, cm_plot_path)

    # Artifact 4: test_evaluation_summary.txt
    summary_path = OUTPUT_DIR / "test_evaluation_summary.txt"
    summary_text = (
        f"========================================================================\n"
        f"FORCORA — DISTILBERT FINAL TEST SET EVALUATION SUMMARY\n"
        f"========================================================================\n"
        f"Checkpoint Evaluated:       {CHECKPOINT_PATH}\n"
        f"Test Dataset Path:          {TEST_DATA_PATH}\n"
        f"Total Test Samples:         {total_test_samples}\n"
        f"Input Text Column:          Review (Raw Text, max_length=256)\n"
        f"Label Column:               Label (3 classes)\n"
        f"Hardware Used:              {gpu_name} (Device: {device})\n"
        f"FP16 Autocast:              {'Enabled' if use_cuda else 'Disabled'}\n"
        f"Total Inference Duration:   {total_duration:.2f} seconds\n"
        f"Average Latency:            {avg_latency_ms:.2f} ms per sample\n"
        f"Peak GPU VRAM Allocated:    {peak_vram_mb:.1f} MiB\n"
        f"------------------------------------------------------------------------\n"
        f"OVERALL PERFORMANCE:\n"
        f"  Test Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n"
        f"  Test Macro-Precision:     {metrics['macro_precision']:.4f}\n"
        f"  Test Macro-Recall:        {metrics['macro_recall']:.4f}\n"
        f"  Test Macro-F1:            {metrics['macro_f1']:.4f} ({metrics['macro_f1'] * 100:.2f}%)\n"
        f"------------------------------------------------------------------------\n"
        f"PER-CLASS CLASSIFICATION REPORT:\n"
        f"{metrics['summary_table']}\n"
        f"------------------------------------------------------------------------\n"
        f"CONFUSION MATRIX (Rows: True, Columns: Predicted):\n"
        f"                Pred 0 (Feature)   Pred 1 (Bug)   Pred 2 (Irrel)    Total\n"
        f"True 0 (Feature)      {cm[0, 0]:<15}    {cm[0, 1]:<12}   {cm[0, 2]:<15}   {cm[0].sum()}\n"
        f"True 1 (Bug)          {cm[1, 0]:<15}    {cm[1, 1]:<12}   {cm[1, 2]:<15}   {cm[1].sum()}\n"
        f"True 2 (Irrel)        {cm[2, 0]:<15}    {cm[2, 1]:<12}   {cm[2, 2]:<15}   {cm[2].sum()}\n"
        f"Total Predicted       {cm[:, 0].sum():<15}    {cm[:, 1].sum():<12}   {cm[:, 2].sum():<15}   {cm.sum()}\n"
        f"========================================================================\n"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Saved: {summary_path}")

    # 8. Print formatted report to console
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
