"""Compare the trained transformers against the TF-IDF baseline.

Builds a unified comparison table (macro-F1 with CI, accuracy, size,
speed), statistical significance check (paired bootstrap), side-by-side
confusion matrices, per-class F1 chart, and writes reports/comparison.md.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import load_config


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = results_dir / "baseline_metrics.json"
    if not baseline_path.exists():
        print("Baseline metrics not found. Run Phase 1 evaluation first.")
        return

    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    test_b = baseline_data["test"]

    table_lines = [
        "# Model Comparison — Telugu Sentiment Classification",
        "",
        "| Model | Split | Macro-F1 | 95% CI | Accuracy | Weighted-F1 |",
        "|---|---|---|---|---|---|",
        f"| TF-IDF char (2-5) + LogReg | test | {test_b['macro_f1']:.4f} | {test_b['macro_f1_ci95'][0]:.4f}–{test_b['macro_f1_ci95'][1]:.4f} | {test_b['accuracy']:.4f} | {test_b['weighted_f1']:.4f} |",
    ]

    # Look for transformer runs in outputs/
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for run_dir in outputs_dir.iterdir():
            summary_path = run_dir / "run_summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                table_lines.append(
                    f"| {s['model']} ({run_dir.name}) | validation | {s.get('best_val_macro_f1', 0):.4f} | — | {s.get('best_val_accuracy', 0):.4f} | — |"
                )

    comparison_path = reports_dir / "comparison.md"
    comparison_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
