"""Tests for stratified splitting: reproducibility, coverage, proportions."""

import pandas as pd

from src.data import stratified_split


def _make_frame(n: int = 600) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = ["neg", "neutral", "pos"][i % 3]
        rows.append({"text": f"వాక్యం {i} {label}", "label": label})
    return pd.DataFrame(rows)


def test_same_seed_gives_identical_splits() -> None:
    df = _make_frame()
    first = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    second = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    for name in ("train", "validation", "test"):
        assert list(first[name]["text"]) == list(second[name]["text"])


def test_different_seed_gives_different_test_split() -> None:
    df = _make_frame()
    first = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    second = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=7)
    assert list(first["test"]["text"]) != list(second["test"]["text"])


def test_splits_partition_the_rows_with_no_overlap() -> None:
    df = _make_frame()
    parts = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    total = sum(len(p) for p in parts.values())
    assert total == len(df)
    ids = pd.concat(parts.values())["text"].tolist()
    assert len(ids) == len(set(ids))


def test_split_sizes_match_ratios() -> None:
    df = _make_frame()
    parts = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    n = len(df)
    assert len(parts["train"]) == round(0.7 * n)
    assert len(parts["validation"]) == round(0.15 * n)
    assert len(parts["test"]) == n - len(parts["train"]) - len(parts["validation"])


def test_splits_are_stratified_by_label() -> None:
    df = _make_frame()
    parts = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    overall = df["label"].value_counts(normalize=True)
    for name, part in parts.items():
        share = part["label"].value_counts(normalize=True)
        for label, expected in overall.items():
            assert abs(share[label] - expected) < 0.05, f"{name}/{label}"
