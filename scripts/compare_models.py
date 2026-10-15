"""Compare the TF-IDF baseline against the fine-tuned transformers.

Builds the comparison table (test macro-F1 with CI, accuracy, weighted-F1,
CPU inference speed, model size), a paired-bootstrap significance check of
the best transformer vs the baseline, side-by-side confusion matrices,
a per-class F1 bar chart, and reports/comparison.md (facts only).

Run AFTER: evaluate_baseline.py (local or Colab) and eval_transformer.py
for every final model. Model size and CPU speed are measured only where a
checkpoint/baseline model file exists; otherwise the table says n/a.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.evaluate import compute_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402


def model_size_mb(directory: Path) -> float | None:
    files = list(directory.glob("**/*")) if directory.exists() else []
    total = sum(f.stat().st_size for f in files if f.is_file())
    return round(total / 1e6, 1) if total else None


def cpu_speed(model_dir: Path, texts: list[str], batch_size: int = 64) -> float | None:
    """Examples per second on CPU for a transformer checkpoint; None if unavailable."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = tokenizer(
                    texts[i : i + batch_size], return_tensors="pt", truncation=True,
                    max_length=64, padding=True,
                )
                model(**inputs)
        return round(len(texts) / (time.time() - start), 1)
    except Exception as exc:  # checkpoint not on this machine, etc.
        print(f"  (CPU speed unavailable for {model_dir.name}: {type(exc).__name__})")
        return None


def paired_bootstrap_delta(
    y_true: list[str], preds_a: list[str], preds_b: list[str],
    labels: list[str], n_resamples: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    """95% CI for macro-F1(A) - macro-F1(B) over resampled test indices."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(labels)
    y_arr = np.array([label2id[y] for y in y_true], dtype=np.int32)
    a_arr = np.array([label2id[p] for p in preds_a], dtype=np.int32)
    b_arr = np.array([label2id[p] for p in preds_b], dtype=np.int32)

    def fast_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
        cm = np.bincount(y * num_classes + pred, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        f1s = np.zeros(num_classes)
        mask = denom > 0
        f1s[mask] = 2 * tp[mask] / denom[mask]
        return float(np.mean(f1s))

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = fast_macro_f1(y_arr[idx], a_arr[idx]) - fast_macro_f1(y_arr[idx], b_arr[idx])
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-model", default=None, help="short name of best model; default = highest val macro-F1 in results/runs/")
    args = parser.parse_args()

    cfg = load_config()
    labels: list[str] = cfg["labels"]
    results_dir = Path(cfg["paths"]["results"])
    reports_dir = Path(cfg["paths"]["reports"])
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = json.loads((results_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    transformer_metrics = {
        p.name.removeprefix("transformer_").removesuffix("_test_metrics.json"): json.loads(p.read_text(encoding="utf-8"))
        for p in results_dir.glob("transformer_*_test_metrics.json")
    }
    if not transformer_metrics:
        raise SystemExit("No transformer_*_test_metrics.json found — run eval_transformer.py per final model first.")

    test_texts = pd.read_csv(Path(cfg["paths"]["processed"]) / "test.csv")["text"].tolist()
    baseline_model_path = results_dir / "baseline_model.joblib"

    baseline_speed = None
    if baseline_model_path.exists():
        import joblib

        baseline = joblib.load(baseline_model_path)
        start = time.time()
        baseline.predict(test_texts)
        baseline_speed = round(len(test_texts) / (time.time() - start), 1)

    # Best model: mean validation macro-F1 across seeds (results/runs), else highest test F1.
    best_name = args.best_model
    if best_name is None:
        runs = list((results_dir / "runs").glob("*.json"))
        if runs:
            by_model: dict[str, list[float]] = {}
            for p in runs:
                s = json.loads(p.read_text(encoding="utf-8"))
                by_model.setdefault(s["model"].split("/")[-1], []).append(s["best_val_macro_f1"])
            best_name = max(by_model, key=lambda m: sum(by_model[m]) / len(by_model[m]))
        else:
            best_name = max(transformer_metrics, key=lambda m: transformer_metrics[m]["macro_f1"])
    print("best model (validation-selected):", best_name)

    rows = [{
        "model": "TF-IDF char(2-5)+LogReg",
        "macro_f1": baseline_metrics["test"]["macro_f1"],
        "ci": baseline_metrics["test"]["macro_f1_ci95"],
        "accuracy": baseline_metrics["test"]["accuracy"],
        "weighted_f1": baseline_metrics["test"]["weighted_f1"],
        "cpu_examples_per_second": baseline_speed,
        "size_mb": model_size_mb(baseline_model_path) if baseline_model_path.exists() else None,
        "confusion_matrix": baseline_metrics["test"]["confusion_matrix"],
        "per_class": baseline_metrics["test"]["per_class"],
    }]
    for name, m in sorted(transformer_metrics.items()):
        ckpt = Path(cfg["paths"]["outputs"]) / f"{name}_seed42"
        rows.append({
            "model": name,
            "macro_f1": m["macro_f1"],
            "ci": m["macro_f1_ci95"],
            "accuracy": m["accuracy"],
            "weighted_f1": m["weighted_f1"],
            "cpu_examples_per_second": cpu_speed(ckpt, test_texts) if ckpt.exists() else None,
            "size_mb": model_size_mb(ckpt / "best_model") if ckpt.exists() else None,
            "confusion_matrix": m["confusion_matrix"],
            "per_class": m["per_class"],
        })

    (results_dir / "comparison_table.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Side-by-side confusion matrices
    fig, axes = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 4.5))
    for ax, row in zip(np.atleast_1d(axes), rows):
        sns_cm = np.array(row["confusion_matrix"])
        im = ax.imshow(sns_cm, cmap="Blues")
        for r in range(sns_cm.shape[0]):
            for c in range(sns_cm.shape[1]):
                ax.text(c, r, sns_cm[r, c], ha="center", va="center",
                        color="white" if sns_cm[r, c] > sns_cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(row["model"], fontsize=10)
    fig.tight_layout()
    fig.savefig(results_dir / "comparison_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(rows)
    for i, row in enumerate(rows):
        f1s = [row["per_class"][lbl]["f1"] for lbl in labels]
        ax.bar([j + i * width for j in range(len(labels))], f1s, width, label=row["model"])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(labels))], labels)
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "comparison_per_class_f1.png", dpi=150)
    plt.close(fig)

    # Paired bootstrap: best transformer vs baseline
    base_preds_df = pd.read_csv(results_dir / "baseline_test_predictions.csv")
    best_preds_df = pd.read_csv(results_dir / f"transformer_{best_name}_test_predictions.csv")
    assert len(base_preds_df) == len(best_preds_df)
    lo, hi = paired_bootstrap_delta(
        base_preds_df["true_label"].tolist(),
        best_preds_df["predicted_label"].tolist(),
        base_preds_df["predicted_label"].tolist(),
        labels,
        seed=cfg["seed"],
    )
    point = transformer_metrics[best_name]["macro_f1"] - baseline_metrics["test"]["macro_f1"]
    significance = (
        f"Paired bootstrap (1000 resamples, seed {cfg['seed']}): {best_name} - baseline "
        f"macro-F1 = {point:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] — "
        + ("difference is statistically significant." if (lo > 0 or hi < 0) else "difference is WITHIN NOISE (CI includes 0).")
    )

    def fmt(v: float | None, suffix: str = "") -> str:
        return f"{v:.1f}{suffix}" if isinstance(v, float) else "n/a"

    lines = [
        "# Model comparison — Telugu sentiment (Phase 2)",
        "",
        "All numbers are measured test-set results; no estimates.",
        "",
        "| Model | Test macro-F1 | 95% CI | Accuracy | Weighted-F1 | CPU ex/s | Size (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['macro_f1']:.4f} | {row['ci'][0]:.4f}–{row['ci'][1]:.4f} | "
            f"{row['accuracy']:.4f} | {row['weighted_f1']:.4f} | "
            f"{fmt(row['cpu_examples_per_second'])} | {fmt(row['size_mb'])} |"
        )
    lines += ["", f"**{significance}**", "",
              "Artifacts: `results/comparison_table.json`, `results/comparison_confusion_matrices.png`, `results/comparison_per_class_f1.png`."]
    out = reports_dir / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
