"""Tests for baseline pipelines and validation-only tuning."""

import pytest

from src.baseline import build_pipeline, tune_baselines


def _synthetic_data():
    """Tiny corpus with class-indicative words so a baseline can score high."""
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    templates = {
        "neg": "చెడు బోరింగ్ సినిమా వద్దు",
        "neutral": "వార్త నివేదిక వివరాలు తెలిశాయి",
        "pos": "మంచి ఆనందం సూపర్ సినిమా",
    }
    for label, template in templates.items():
        for i in range(30):
            train_texts.append(f"{template} {i}")
            train_labels.append(label)
        for i in range(30, 42):
            val_texts.append(f"{template} {i}")
            val_labels.append(label)
    return train_texts, train_labels, val_texts, val_labels


def test_build_pipeline_word_configuration() -> None:
    pipe = build_pipeline("word", ngram_range=(1, 2), min_df=2, C=1.0, seed=42)
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    assert tfidf.analyzer == "word"
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.min_df == 2
    assert clf.C == 1.0
    assert clf.class_weight == "balanced"


def test_build_pipeline_char_uses_char_wb() -> None:
    pipe = build_pipeline("char", ngram_range=(2, 5), min_df=1, C=0.1, seed=42)
    assert pipe.named_steps["tfidf"].analyzer == "char_wb"
    assert pipe.named_steps["tfidf"].ngram_range == (2, 5)


def test_build_pipeline_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        build_pipeline("byte", ngram_range=(1, 1), min_df=1, C=1.0, seed=42)


def test_tune_baselines_finds_strong_model_on_synthetic_data() -> None:
    train_x, train_y, val_x, val_y = _synthetic_data()
    result = tune_baselines(train_x, train_y, val_x, val_y, seed=42)
    assert result["best"]["val_macro_f1"] >= 0.9
    assert result["best"]["variant"] in {"word", "char"}
    for key in ("variant", "ngram_range", "min_df", "C", "val_macro_f1"):
        assert key in result["best"]
    # Every tried combination is reported, without the model objects.
    assert len(result["all_results"]) == 24
    assert all("model" not in row for row in result["all_results"])


def test_tune_baselines_is_deterministic_given_seed() -> None:
    train_x, train_y, val_x, val_y = _synthetic_data()
    first = tune_baselines(train_x, train_y, val_x, val_y, seed=42)
    second = tune_baselines(train_x, train_y, val_x, val_y, seed=42)
    assert first["best"]["val_macro_f1"] == second["best"]["val_macro_f1"]
    assert first["best"]["ngram_range"] == second["best"]["ngram_range"]
    assert first["best"]["C"] == second["best"]["C"]
