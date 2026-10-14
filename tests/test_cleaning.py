"""Tests for text cleaning (NFC normalization, whitespace, empty removal)."""

import unicodedata

import pandas as pd

from src.data import clean_dataframe, clean_text


def test_clean_text_applies_nfc_normalization() -> None:
    # U+0C48 (ై) is one of the few Telugu vowel signs with a canonical
    # decomposition, so NFD really changes this string.
    nfc = "నీళ్లు పైకి వచ్చాయి"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc  # guard: the decomposed form really is different
    assert clean_text(nfd) == nfc


def test_clean_text_collapses_and_strips_whitespace() -> None:
    messy = "  అది   చాలా\n\t మంచి   సినిమా  "
    assert clean_text(messy) == "అది చాలా మంచి సినిమా"


def test_clean_text_keeps_english_and_digits() -> None:
    assert clean_text("T20 వరల్డ్ కప్ 2024") == "T20 వరల్డ్ కప్ 2024"


def test_clean_dataframe_drops_empty_and_null_texts() -> None:
    df = pd.DataFrame(
        {"text": ["మంచి సినిమా", "", "   ", None], "label": ["pos", "neg", "neg", "pos"]}
    )
    out = clean_dataframe(df)
    assert len(out) == 1
    assert out.iloc[0]["text"] == "మంచి సినిమా"
    assert out.iloc[0]["label"] == "pos"
