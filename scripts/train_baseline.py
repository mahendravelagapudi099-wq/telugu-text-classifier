"""Train and tune the TF-IDF + Logistic Regression baseline.

Tunes on train+validation only, saves the best model (joblib), its
config, and the full grid results to results/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.baseline import tune_baselines  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "validation.csv")

    tuned = tune_baselines(
        train["text"].tolist(),
        train["label"].tolist(),
        val["text"].tolist(),
        val["label"].tolist(),
        seed=cfg["seed"],
    )

    model_path = results_dir / "baseline_model.joblib"
    joblib.dump(tuned["best_model"], model_path)

    payload = {
        "dataset": cfg["dataset"]["name"],
        "seed": cfg["seed"],
        "best": tuned["best"],
        "all_results": tuned["all_results"],
    }
    out_path = results_dir / "baseline_tuning.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"best: {json.dumps(tuned['best'])}")
    print(f"saved model -> {model_path}")
    print(f"saved tuning results -> {out_path}")


if __name__ == "__main__":
    main()
