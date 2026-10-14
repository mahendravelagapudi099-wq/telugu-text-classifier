"""Tests for exact-duplicate and near-duplicate (leakage) removal."""

import pandas as pd

from src.data import (
    clean_dataframe,
    find_near_duplicate_groups,
    jaccard,
    remove_exact_duplicates,
    remove_near_duplicates,
    shingles,
    stratified_split,
)


def test_shingles_returns_character_windows() -> None:
    got = shingles("అంది", size=2)
    assert got == {"అం", "ంద", "ది"}


def test_jaccard_on_known_sets() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert jaccard({"a", "b", "c"}, {"a", "b", "d"}) == 0.5


def test_remove_exact_duplicates_keeps_first_occurrence() -> None:
    df = pd.DataFrame(
        {"text": ["నవ్వు", "బాధ", "నవ్వు", "నవ్వు"], "label": ["pos", "neg", "neg", "neutral"]}
    )
    out, removed = remove_exact_duplicates(df)
    assert removed == 2
    assert len(out) == 2
    # First occurrence wins, including its label.
    assert list(out["text"]) == ["నవ్వు", "బాధ"]
    assert list(out["label"]) == ["pos", "neg"]


def test_near_duplicate_groups_catches_one_char_edit() -> None:
    a = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమే"
    b = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమా"
    groups = find_near_duplicate_groups([a, b], shingle_size=5, threshold=0.8)
    assert {0, 1} in [set(g) for g in groups]


def test_near_duplicate_groups_separates_different_texts() -> None:
    a = "క్రికెట్ జట్టు ఫైనల్ కు చేరుకుంది అందరూ ఆనందంతో ఆడిపడ్డారు"
    b = "వర్షాల కారణంగా రైతులు ఈ ఏడాది మంచి పంట పొందారుట అని అధికారులు చెప్పారు"
    groups = find_near_duplicate_groups([a, b], shingle_size=5, threshold=0.8)
    assert groups == []


def test_remove_near_duplicates_reports_and_keeps_one_per_group() -> None:
    a = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమే"
    b = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమా"
    c = "వేరే వాక్యం ఇది పూర్తిగా భిన్నమైన అంశం గురించి మాట్లాడుతుంది"
    df = pd.DataFrame({"text": [a, b, c], "label": ["pos", "neg", "neutral"]})
    out, stats = remove_near_duplicates(df, shingle_size=5, threshold=0.8)
    assert stats["near_duplicates_removed"] == 1
    assert list(out["text"]) == [a, c]
    assert list(out["label"]) == ["pos", "neutral"]


def test_pipeline_leaves_no_near_duplicates_across_splits() -> None:
    """Dedup before splitting must guarantee no near-dup spans two splits."""
    a = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమే"
    b = "ఈ సినిమా చాలా మంచి సినిమాగా అందరూ చెబుతున్నారు నిజమా"
    labels = ["neg", "neutral", "pos"]
    topics = ["క్రికెట్", "వర్షం", "సినిమా", "ఎన్నికలు", "వ్యవసాయం", "పరీక్షలు", "పండగ", "రోడ్డు"]
    places = ["హైదరాబాద్", "విజయవాడ", "విశాఖ", "గుంటూరు", "తిరుపతి", "వరంగల్", "ఖమ్మం"]
    rows = [
        {"text": t, "label": labels[i % 3]}
        for i, t in enumerate(
            [a, b]
            + [
                f"{topics[i % 8]} వార్తలు {i} తేదీన {places[i % 7]} ప్రాంతంలో చర్చ జరిగింది"
                for i in range(30)
            ]
        )
    ]
    df = clean_dataframe(pd.DataFrame(rows))
    df, _ = remove_exact_duplicates(df)
    df, _ = remove_near_duplicates(df, shingle_size=5, threshold=0.8)
    parts = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)

    pooled = pd.concat(parts.values())["text"].tolist()
    assert find_near_duplicate_groups(pooled, shingle_size=5, threshold=0.8) == []
