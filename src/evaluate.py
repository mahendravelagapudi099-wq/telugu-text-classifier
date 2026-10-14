"""Evaluation: metrics, confidence intervals, confusion matrices, error export.

Macro-F1 is the primary metric: with imbalanced classes (neutral 54%,
neg 19%) accuracy and weighted-F1 are dominated by the majority class,
while macro-F1 weights every class equally.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, Any]:
    """Return accuracy, macro/weighted F1, per-class metrics, confusion matrix.

    Confusion matrix rows are true labels, columns are predicted labels,
    in the order of `labels`.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile CI for macro-F1 via bootstrap resampling of indices."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_true = [y_true[j] for j in idx]
        sample_pred = [y_pred[j] for j in idx]
        scores[i] = f1_score(
            sample_true, sample_pred, labels=labels, average="macro", zero_division=0
        )
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_confusion_matrix(
    matrix: list[list[int]], labels: list[str], out_path: str | Path
) -> Path:
    """Render the confusion matrix as a labeled heatmap PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.array(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_errors(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    confidence: list[float],
    out_path: str | Path,
) -> Path:
    """Write all misclassified examples to CSV for manual error categorization.

    Columns: text, true_label, predicted_label, confidence, and an empty
    my_error_category column to be filled in by hand.
    """
    rows = [
        {"text": t, "true_label": tr, "predicted_label": pr, "confidence": c}
        for t, tr, pr, c in zip(texts, y_true, y_pred, confidence)
        if tr != pr
    ]
    df = pd.DataFrame(
        rows, columns=["text", "true_label", "predicted_label", "confidence", "my_error_category"]
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def predict_with_confidence(
    model: Any, texts: list[str]
) -> tuple[list[str], list[float]]:
    """Return model predictions and their confidence (max class probability)."""
    preds = model.predict(texts)
    proba = model.predict_proba(texts)
    return list(preds), [float(p.max()) for p in proba]
