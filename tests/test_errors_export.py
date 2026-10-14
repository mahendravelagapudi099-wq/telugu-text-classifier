"""Tests for misclassified-example export."""

import pandas as pd

from src.evaluate import export_errors


def test_export_contains_only_errors_with_required_columns() -> None:
    texts = ["మంచి సినిమా", "చెడు వార్త", "సామాన్య విషయం", "గెలుపు"]
    y_true = ["pos", "neg", "neutral", "pos"]
    y_pred = ["pos", "neutral", "neutral", "neg"]
    confidence = [0.9, 0.4, 0.8, 0.3]
    out = export_errors(texts, y_true, y_pred, confidence, "results/errors_test.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "text",
        "true_label",
        "predicted_label",
        "confidence",
        "my_error_category",
    ]
    assert len(df) == 2  # only the two misclassified rows
    assert set(df["text"]) == {"చెడు వార్త", "గెలుపు"}
    assert df["my_error_category"].isna().all()
