"""Load, clean, and split the Telugu classification dataset.

Cleaning pipeline: NFC-normalize and collapse whitespace, drop empty texts,
drop exact duplicates, then drop near-duplicates (character-shingle
Jaccard) BEFORE splitting so no near-duplicate text can span two splits.
"""

import unicodedata
from collections import defaultdict
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split


class DedupStats(TypedDict):
    """Counts reported by near-duplicate removal."""

    near_duplicate_groups: int
    near_duplicates_removed: int
    shingle_size: int
    jaccard_threshold: float


def clean_text(text: str) -> str:
    """NFC-normalize, collapse whitespace runs, and strip the ends."""
    normalized = unicodedata.normalize("NFC", str(text))
    return " ".join(normalized.split())


def clean_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Clean every text and drop rows that are null or empty after cleaning."""
    out = df.copy()
    out[text_col] = out[text_col].map(lambda t: clean_text(t) if pd.notna(t) else "")
    return out[out[text_col] != ""].reset_index(drop=True)


def remove_exact_duplicates(df: pd.DataFrame, text_col: str = "text") -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate texts, keeping the first occurrence.

    Returns (deduplicated frame, number of rows removed).
    """
    dup_mask = df[text_col].duplicated(keep="first")
    removed = int(dup_mask.sum())
    return df.loc[~dup_mask].reset_index(drop=True), removed


def shingles(text: str, size: int = 5) -> set[str]:
    """Return the set of character shingles of `text`."""
    if len(text) <= size:
        return {text}
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets; empty==empty counts as identical."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicate_groups(
    texts: list[str],
    shingle_size: int = 5,
    threshold: float = 0.8,
    max_block: int = 100,
) -> list[list[int]]:
    """Find groups of near-duplicate texts (Jaccard >= threshold).

    Candidate pairs come from an inverted index over shingles; shingles
    occurring in more than `max_block` texts are skipped because highly
    frequent shingles cannot distinguish duplicates. Matches are merged
    into groups with union-find, so groups are deterministic and every
    group is keyed by its lowest index.
    """
    n = len(texts)
    shingle_sets = [shingles(t, shingle_size) for t in texts]
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(shingle_sets):
        for sh in s:
            index[sh].append(i)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen_pairs: set[tuple[int, int]] = set()
    for rows in index.values():
        if len(rows) > max_block:
            continue
        for x in range(len(rows)):
            for y in range(x + 1, len(rows)):
                pair = (rows[x], rows[y])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if jaccard(shingle_sets[pair[0]], shingle_sets[pair[1]]) >= threshold:
                    union(*pair)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def remove_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    shingle_size: int = 5,
    threshold: float = 0.8,
    max_block: int = 100,
) -> tuple[pd.DataFrame, DedupStats]:
    """Keep only the first member of each near-duplicate group.

    Returns (deduplicated frame, stats dict with removal counts).
    """
    groups = find_near_duplicate_groups(
        df[text_col].tolist(), shingle_size, threshold, max_block
    )
    drop_positions = sorted(i for g in groups for i in g[1:])
    out = df.drop(index=df.index[drop_positions]).reset_index(drop=True)
    stats: DedupStats = {
        "near_duplicate_groups": len(groups),
        "near_duplicates_removed": len(drop_positions),
        "shingle_size": shingle_size,
        "jaccard_threshold": threshold,
    }
    return out, stats


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Deterministically split into train/validation/test, stratified by label."""
    train_ratio, val_ratio, _ = ratios
    # Integer counts: float fractions like 0.15/0.3 carry representation
    # error (0.4999...) which sklearn floors, silently losing a row.
    n_train = int(round(train_ratio * len(df)))
    n_val = int(round(val_ratio * len(df)))
    train, temp = train_test_split(
        df, train_size=n_train, random_state=seed, stratify=df[label_col]
    )
    validation, test = train_test_split(
        temp, train_size=n_val, random_state=seed, stratify=temp[label_col]
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
