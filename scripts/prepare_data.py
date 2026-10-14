"""Download the approved dataset and build cleaned, leakage-free, fixed splits.

Writes data/processed/{train,validation,test}.csv and a run summary to
results/data_summary.json. Every later model must use these exact files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

from src.data import (  # noqa: E402
    clean_dataframe,
    remove_exact_duplicates,
    remove_near_duplicates,
    stratified_split,
)
from src.utils import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    ds_cfg, dedup_cfg = cfg["dataset"], cfg["dedup"]
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(ds_cfg["name"])
    frames = [dataset[split].to_pandas() for split in ("train", "validation", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = df[[ds_cfg["text_column"], ds_cfg["label_column"]]]
    df.columns = ["text", "label"]
    n_raw = len(df)

    actual_labels = set(df["label"].dropna().unique())
    if actual_labels != set(cfg["labels"]):
        raise ValueError(f"Labels {sorted(actual_labels)} != config {sorted(cfg['labels'])}")

    df = clean_dataframe(df)
    n_clean = len(df)
    df, n_exact_dupes = remove_exact_duplicates(df)
    df, near_stats = remove_near_duplicates(
        df,
        shingle_size=dedup_cfg["shingle_size"],
        threshold=dedup_cfg["jaccard_threshold"],
    )

    ratios = (
        cfg["splits"]["train"],
        cfg["splits"]["validation"],
        cfg["splits"]["test"],
    )
    parts = stratified_split(df, ratios=ratios, seed=cfg["seed"])
    for name, part in parts.items():
        part.to_csv(processed / f"{name}.csv", index=False)

    summary = {
        "dataset": ds_cfg["name"],
        "rows_raw": n_raw,
        "empty_or_null_removed": n_raw - n_clean,
        "exact_duplicates_removed": n_exact_dupes,
        **near_stats,
        "rows_final": len(df),
        "split_sizes": {name: len(part) for name, part in parts.items()},
        "split_ratios": list(ratios),
        "seed": cfg["seed"],
    }
    summary_path = results_dir / "data_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
