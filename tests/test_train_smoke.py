"""Smoke tests: the transformer training script runs on CPU and reloads.

Uses 50 examples, 1 epoch, one of the approved models. The ~1.1GB model
download happens once and is cached by the Hugging Face hub.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_transformer.py"
MODEL = "ai4bharat/IndicBERTv2-MLM-only"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_transformer", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_weights_are_inverse_frequency() -> None:
    module = _load_script()
    labels = ["neg", "neutral", "pos"]
    counts = {"neg": 10, "neutral": 80, "pos": 40}
    weights = module.compute_class_weights(labels, counts)
    # Most frequent class gets the smallest weight; ratio neg/pos == 4.0
    assert weights[labels.index("neutral")] < weights[labels.index("pos")]
    assert weights[labels.index("pos")] < weights[labels.index("neg")]
    assert weights[labels.index("neg")] / weights[labels.index("pos")] == pytest.approx(4.0)


def test_training_smoke_run_saves_and_reloads_checkpoint(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            MODEL,
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-length",
            "64",
            "--train-limit",
            "50",
            "--val-limit",
            "50",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    run_dir = tmp_path / "run"
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == MODEL
    assert 0.0 <= summary["best_val_macro_f1"] <= 1.0
    assert summary["training_seconds"] > 0
    assert (run_dir / "loss_curves.png").exists()

    checkpoint_dirs = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
    assert checkpoint_dirs, "no checkpoint saved"
    assert (run_dir / "best_model").is_dir(), "best_model directory was not saved"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(run_dir / "best_model"))
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        str(run_dir / "best_model")
    )
    import torch

    inputs = tokenizer(["మంచి సినిమా", "చెడు వార్త"], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = reloaded(**inputs).logits
    assert logits.shape == (2, 3)
    assert {int(i) for i in logits.argmax(-1)} <= {0, 1, 2}
