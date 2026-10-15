"""Evaluate a fine-tuned transformer checkpoint on the held-out test set.

Called once per final model. Reuses the Phase 1 evaluation framework:
macro-F1 with bootstrap CI, confusion matrix PNG, misclassified-examples
CSV. Writes results/transformer_<name>_test_metrics.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.evaluate import (  # noqa: E402
    bootstrap_macro_f1,
    compute_metrics,
    export_errors,
    plot_confusion_matrix,
)
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="dir with best_model/")
    parser.add_argument("--name", required=True, help="short name for output files")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    labels: list[str] = cfg["labels"]
    processed = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(processed / "test.csv")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_dir / "best_model")
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "best_model"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions: list[str] = []
    confidences: list[float] = []
    inference_start = time.time()
    with torch.no_grad():
        for i in range(0, len(test_df), args.batch_size):
            batch = test_df["text"].iloc[i : i + args.batch_size].tolist()
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=cfg["training"]["max_length"],
                padding=True,
            ).to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            confidences += probs.max(dim=-1).values.tolist()
            predictions += [labels[j] for j in probs.argmax(dim=-1).tolist()]
    inference_seconds = time.time() - inference_start

    y_true = test_df["label"].tolist()
    metrics = compute_metrics(y_true, predictions, labels=labels)
    ci_low, ci_high = bootstrap_macro_f1(
        y_true, predictions, labels=labels, n_resamples=1000, seed=cfg["seed"]
    )
    metrics["macro_f1_ci95"] = [ci_low, ci_high]

    plot_confusion_matrix(
        metrics["confusion_matrix"], labels, results_dir / f"confusion_matrix_{args.name}_test.png"
    )
    export_errors(
        test_df["text"].tolist(), y_true, predictions, confidences,
        results_dir / f"errors_{args.name}_test.csv",
    )
    # Full predictions: needed for paired bootstrap against the baseline
    pd.DataFrame(
        {"text": test_df["text"], "true_label": y_true,
         "predicted_label": predictions, "confidence": confidences}
    ).to_csv(results_dir / f"transformer_{args.name}_test_predictions.csv", index=False)

    payload = {
        "model": args.name,
        "model_dir": str(args.model_dir),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "test_rows": len(test_df),
        "inference_seconds": round(inference_seconds, 2),
        "examples_per_second": round(len(test_df) / inference_seconds, 1),
        "macro_f1": metrics["macro_f1"],
        "macro_f1_ci95": metrics["macro_f1_ci95"],
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    out_path = results_dir / f"transformer_{args.name}_test_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{args.name}: test macro-F1 {metrics['macro_f1']:.4f} "
        f"(95% CI {ci_low:.4f}-{ci_high:.4f}), accuracy {metrics['accuracy']:.4f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
