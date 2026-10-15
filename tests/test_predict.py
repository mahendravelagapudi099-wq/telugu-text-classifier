"""Tests for the inference module src/predict.py."""

import pytest
from src.predict import TeluguSentimentPredictor

@pytest.fixture(scope="module")
def predictor():
    # Initializes using the local baseline model or available weights
    return TeluguSentimentPredictor(threshold=0.55)

def test_predict_single_text(predictor):
    res = predictor.predict("సినిమా చాలా అద్భుతంగా ఉంది.")
    assert isinstance(res, dict)
    assert "label" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert set(res["probabilities"].keys()) == {"neg", "neutral", "pos"}
    assert res["label"] in ["neg", "neutral", "pos", "uncertain"]

def test_predict_empty_and_whitespace_input(predictor):
    empty_res = predictor.predict("")
    assert empty_res["label"] == "uncertain"
    assert empty_res["confidence"] == 0.0
    assert empty_res["is_uncertain"] is True

    space_res = predictor.predict("   \n\t  ")
    assert space_res["label"] == "uncertain"
    assert space_res["confidence"] == 0.0

def test_predict_very_long_input_truncation(predictor):
    long_text = "సినిమా చాలా బాగుంది " * 200  # Over 600 words
    res = predictor.predict(long_text)
    assert isinstance(res, dict)
    assert res["label"] in ["neg", "neutral", "pos", "uncertain"]

def test_predict_non_telugu_text(predictor):
    english_text = "This is a purely English headline about tech and business."
    res = predictor.predict(english_text)
    assert isinstance(res, dict)
    assert "label" in res
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)

def test_predict_mixed_script_text(predictor):
    mixed_text = "ఈ movie చాలా super గా ఉంది bro!"
    res = predictor.predict(mixed_text)
    assert isinstance(res, dict)
    assert "label" in res

def test_predict_batching(predictor):
    texts = [
        "అద్భుతమైన విజయం.",
        "చాలా దారుణమైన పరిస్థితి.",
        "రేపు ఉదయం సమావేశం జరుగుతుంది.",
        "",
        "Mixed review but overall okay.",
    ]
    batch_res = predictor.predict(texts, batch_size=2)
    assert isinstance(batch_res, list)
    assert len(batch_res) == len(texts)
    assert batch_res[3]["is_uncertain"] is True

def test_confidence_threshold_uncertainty(predictor):
    # If threshold is artificially set to 0.99, almost all predictions become uncertain
    strict_predictor = TeluguSentimentPredictor(threshold=0.99)
    res = strict_predictor.predict("సాధారణ వార్త")
    if res["confidence"] < 0.99:
        assert res["label"] == "uncertain"
        assert res["is_uncertain"] is True
