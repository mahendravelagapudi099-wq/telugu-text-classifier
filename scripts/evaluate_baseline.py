"""Evaluate the saved baseline on validation and the held-out test set.

Saves results/baseline_metrics.json (metrics + bootstrap CIs),
results/confusion_matrix_test.png, and results/errors_test.csv for
manual error categorization. This is the only place the test set is used.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import (  # noqa: E402
    bootstrap_macro_f1,
    compute_metrics,
    export_errors,
    plot_confusion_matrix,
    predict_with_confidence,
)
from src.utils import load_config  # noqa: E402


def evaluate_split(model, df: pd.DataFrame, labels: list[str], seed: int) -> dict:
    """Compute full metrics + macro-F1 CI for one split."""
    preds, confidence = predict_with_confidence(model, df["text"].tolist())
    metrics = compute_metrics(df["label"].tolist(), preds, labels=labels)
    ci_low, ci_high = bootstrap_macro_f1(
        df["label"].tolist(), preds, labels=labels, n_resamples=1000, seed=seed
    )
    metrics["macro_f1_ci95"] = [ci_low, ci_high]
    metrics["confidence"] = confidence
    metrics["predictions"] = preds
    return metrics


def main() -> None:
    cfg = load_config()
    labels = cfg["labels"]
    processed = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    model = joblib.load(results_dir / "baseline_model.joblib")

    outputs: dict[str, dict] = {}
    for split in ("validation", "test"):
        df = pd.read_csv(processed / f"{split}.csv")
        outputs[split] = evaluate_split(model, df, labels, cfg["seed"])

    test = outputs["test"]
    plot_confusion_matrix(
        test["confusion_matrix"], labels, results_dir / "confusion_matrix_test.png"
    )
    export_errors(
        pd.read_csv(processed / "test.csv")["text"].tolist(),
        pd.read_csv(processed / "test.csv")["label"].tolist(),
        test["predictions"],
        test["confidence"],
        results_dir / "errors_test.csv",
    )

    serializable = {
        split: {k: v for k, v in out.items() if k not in ("predictions", "confidence")}
        for split, out in outputs.items()
    }
    payload = {
        "model": "tfidf+logreg",
        "dataset": cfg["dataset"]["name"],
        "seed": cfg["seed"],
        **serializable,
    }
    out_path = results_dir / "baseline_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for split, out in outputs.items():
        lo, hi = out["macro_f1_ci95"]
        print(
            f"{split}: macro-F1 {out['macro_f1']:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
            f"accuracy {out['accuracy']:.4f}, weighted-F1 {out['weighted_f1']:.4f}"
        )
    print(f"wrote {out_path}, confusion_matrix_test.png, errors_test.csv")


if __name__ == "__main__":
    main()
