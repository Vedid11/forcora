"""Evaluation metrics computation for multi-class privacy review classification."""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Compute comprehensive class-aware classification metrics.
    
    Calculates:
    - Overall Accuracy
    - Macro Precision, Macro Recall, Macro F1
    - Per-class Precision, Recall, F1, and Support
    - 2D Confusion Matrix [num_classes x num_classes]
    
    Safe from external dependencies like scikit-learn, with built-in zero-division guarding.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if class_names is None:
        class_names = {i: f"Class_{i}" for i in range(num_classes)}

    # Construct Confusion Matrix: C[true, pred]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total_samples = len(y_true)
    correct_samples = int(np.trace(cm))
    accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

    per_class_metrics: Dict[int, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class_metrics[i] = {
            "name": class_names.get(i, f"Class_{i}"),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Format human-readable table string
    table_lines = [
        f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
        "-" * 72,
    ]
    for i in range(num_classes):
        m = per_class_metrics[i]
        table_lines.append(
            f"{m['name']:<22} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8}"
        )
    table_lines.append("-" * 72)
    table_lines.append(
        f"{'Macro Average':<22} | {macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_f1:<10.4f} | {total_samples:<8}"
    )
    table_lines.append(f"{'Overall Accuracy':<22} | {accuracy:<10.4f}")

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "summary_table": "\n".join(table_lines),
    }
