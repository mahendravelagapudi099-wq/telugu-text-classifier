"""Export error-analysis files for the best model (validation-selected).

Produces, under results/error_analysis_<best>/:
- 100_errors.csv: 100 misclassified test examples, stratified across true
  classes, with transformer confidence, the baseline's prediction on the
  same example, and an empty my_error_category column.
- transformer_right_baseline_wrong.csv: 30 examples (seeded sample).
- baseline_right_transformer_wrong.csv: 30 examples (seeded sample).

No error categories are invented here — the human labels them by hand.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.utils import load_config  # noqa: E402

COLUMNS = [
    "text", "true_label", "transformer_prediction", "transformer_confidence",
    "baseline_prediction", "my_error_category",
]


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results"])

    runs = list((results_dir / "runs").glob("*.json"))
    if runs:
        by_model: dict[str, list[float]] = {}
        for p in runs:
            s = json.loads(p.read_text(encoding="utf-8"))
            by_model.setdefault(s["model"].split("/")[-1], []).append(s["best_val_macro_f1"])
        best = max(by_model, key=lambda m: sum(by_model[m]) / len(by_model[m]))
        print("best model (mean validation macro-F1):", best)
    else:
        metrics = {
            p.name.removeprefix("transformer_").removesuffix("_test_metrics.json"):
                json.loads(p.read_text(encoding="utf-8"))["macro_f1"]
            for p in results_dir.glob("transformer_*_test_metrics.json")
        }
        if not metrics:
            raise SystemExit("No transformer test metrics found — nothing to export.")
        best = max(metrics, key=metrics.get)
        print("best model (highest test macro-F1):", best)

    t_preds = pd.read_csv(results_dir / f"transformer_{best}_test_predictions.csv")
    b_preds = pd.read_csv(results_dir / "baseline_test_predictions.csv")
    assert len(t_preds) == len(b_preds), "prediction files must cover the same test rows"
    df = pd.DataFrame({
        "text": t_preds["text"],
        "true_label": t_preds["true_label"],
        "transformer_prediction": t_preds["predicted_label"],
        "transformer_confidence": t_preds["confidence"],
        "baseline_prediction": b_preds["predicted_label"],
    })

    out_dir = results_dir / f"error_analysis_{best}"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = df[df["true_label"] != df["transformer_prediction"]]
    stratified_groups = []
    for _, group in errors.groupby("true_label"):
        n_sample = min(len(group), max(1, round(100 * len(group) / len(errors))))
        stratified_groups.append(group.sample(n=n_sample, random_state=cfg["seed"]))
    stratified = pd.concat(stratified_groups) if stratified_groups else errors
    if len(stratified) > 100:
        stratified = stratified.sample(n=100, random_state=cfg["seed"])
    stratified.assign(my_error_category="")[COLUMNS].to_csv(out_dir / "100_errors.csv", index=False)

    t_right = df[(df["true_label"] == df["transformer_prediction"]) & (df["true_label"] != df["baseline_prediction"])]
    t_right.sample(n=min(30, len(t_right)), random_state=cfg["seed"]).assign(
        my_error_category="")[COLUMNS].to_csv(out_dir / "transformer_right_baseline_wrong.csv", index=False)

    b_right = df[(df["true_label"] == df["baseline_prediction"]) & (df["true_label"] != df["transformer_prediction"])]
    b_right.sample(n=min(30, len(b_right)), random_state=cfg["seed"]).assign(
        my_error_category="")[COLUMNS].to_csv(out_dir / "baseline_right_transformer_wrong.csv", index=False)

    print(f"wrote {out_dir}/100_errors.csv ({len(stratified)} rows), "
          f"transformer_right_baseline_wrong.csv ({min(30, len(t_right))}), "
          f"baseline_right_transformer_wrong.csv ({min(30, len(b_right))})")


if __name__ == "__main__":
    main()
