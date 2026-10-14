"""Tests for metric computation on a tiny hand-made example."""

import numpy as np

from src.evaluate import bootstrap_macro_f1, compute_metrics

# Hand-computed example (label order: neg, neutral, pos):
# y_true = [pos, pos, neg, neg, neutral]
# y_pred = [pos, neg, neg, neg, neutral]
# pos:      P=1.0   R=0.5  F1=2*1*0.5/(1+0.5)=0.666667
# neg:      P=2/3   R=1.0  F1=2*(2/3)*1/(2/3+1)=0.8
# neutral:  P=R=F1=1.0
# accuracy = 4/5; macro-F1 = (0.666667+0.8+1.0)/3 = 0.822222
# weighted-F1 = (2*0.666667+2*0.8+1*1.0)/5 = 0.786667
Y_TRUE = ["pos", "pos", "neg", "neg", "neutral"]
Y_PRED = ["pos", "neg", "neg", "neg", "neutral"]
LABELS = ["neg", "neutral", "pos"]


def test_accuracy_macro_and_weighted_f1_match_hand_computation() -> None:
    m = compute_metrics(Y_TRUE, Y_PRED, labels=LABELS)
    assert m["accuracy"] == 0.8
    assert abs(m["macro_f1"] - 0.822222) < 1e-5
    assert abs(m["weighted_f1"] - 0.786667) < 1e-5


def test_per_class_metrics_match_hand_computation() -> None:
    m = compute_metrics(Y_TRUE, Y_PRED, labels=LABELS)
    per_class = m["per_class"]
    assert abs(per_class["pos"]["f1"] - 2 / 3) < 1e-5
    assert abs(per_class["neg"]["f1"] - 0.8) < 1e-5
    assert per_class["neutral"]["f1"] == 1.0
    assert per_class["pos"]["precision"] == 1.0
    assert per_class["pos"]["recall"] == 0.5


def test_confusion_matrix_layout_is_true_rows_pred_cols() -> None:
    m = compute_metrics(Y_TRUE, Y_PRED, labels=LABELS)
    expected = np.array([[2, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert np.array_equal(np.array(m["confusion_matrix"]), expected)


def test_bootstrap_ci_on_perfect_predictions_is_degenerate() -> None:
    y = ["neg", "neutral", "pos"] * 20
    lo, hi = bootstrap_macro_f1(y, y, labels=LABELS, n_resamples=200, seed=42)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic() -> None:
    m = compute_metrics(Y_TRUE, Y_PRED, labels=LABELS)
    lo1, hi1 = bootstrap_macro_f1(Y_TRUE, Y_PRED, labels=LABELS, n_resamples=500, seed=42)
    lo2, hi2 = bootstrap_macro_f1(Y_TRUE, Y_PRED, labels=LABELS, n_resamples=500, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= m["macro_f1"] <= hi1
    assert 0.0 <= lo1 <= hi1 <= 1.0
