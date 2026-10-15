"""Inference module for Telugu Sentiment Classification.

Provides fast, batchable predictions, class probabilities, and confidence
thresholding (flagging predictions as 'uncertain' when confidence is low).
Supports fallback to baseline pipeline when transformer checkpoint is local.
"""

from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
import torch

from src.utils import load_config

DEFAULT_CONFIDENCE_THRESHOLD = 0.50

class TeluguSentimentPredictor:
    """Predictor for Telugu Sentiment Classification with uncertainty thresholding."""

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str | None = None,
        max_length: int = 64,
    ) -> None:
        self.cfg = load_config()
        self.labels: list[str] = self.cfg["labels"]
        self.threshold: float = threshold
        self.max_length: int = max_length

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.baseline_model = None

        # Determine model source
        project_root = Path(__file__).resolve().parents[1]
        
        # Check candidate locations for transformer checkpoint
        checkpoint_dir = None
        if model_path:
            p = Path(model_path)
            if p.exists():
                checkpoint_dir = p
        
        if checkpoint_dir is None:
            candidates = [
                project_root / "outputs/IndicBERTv2-MLM-only_seed123/best_model",
                project_root / "outputs/IndicBERTv2-MLM-only_seed42/best_model",
                project_root / "outputs/IndicBERTv2-MLM-only_seed7/best_model",
            ]
            for c in candidates:
                if c.exists() and (c / "config.json").exists():
                    checkpoint_dir = c
                    break

        if checkpoint_dir is not None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
            self.model.to(self.device).eval()
            self.model_type = "transformer"
            self.model_name = checkpoint_dir.parent.name
        else:
            # Fall back to baseline TF-IDF model if transformer weights are not downloaded locally
            baseline_path = project_root / "results/baseline_model.joblib"
            if baseline_path.exists():
                import joblib
                self.baseline_model = joblib.load(baseline_path)
                self.model_type = "baseline"
                self.model_name = "TF-IDF + LogisticRegression"
            else:
                # Fall back to HF Hub directly
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                hub_id = "ai4bharat/IndicBERTv2-MLM-only"
                self.tokenizer = AutoTokenizer.from_pretrained(hub_id)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    hub_id, num_labels=len(self.labels)
                )
                self.model.to(self.device).eval()
                self.model_type = "transformer"
                self.model_name = hub_id

    def _sanitize_input(self, text: Any) -> str:
        """Handle null, non-string, or empty inputs gracefully."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    def predict_proba(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Return class probability distributions for a text or batch of texts."""
        is_single = isinstance(texts, str)
        raw_list = [texts] if is_single else list(texts)
        sanitized = [self._sanitize_input(t) for t in raw_list]

        all_probs: List[Dict[str, float]] = []

        if self.model_type == "transformer":
            with torch.no_grad():
                for i in range(0, len(sanitized), batch_size):
                    batch_texts = sanitized[i : i + batch_size]
                    
                    # Replace empty texts with whitespace so tokenizer does not fail
                    proc_batch = [t if t else " " for t in batch_texts]

                    inputs = self.tokenizer(
                        proc_batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                    for idx, orig_t in enumerate(batch_texts):
                        if not orig_t:
                            # Empty or whitespace-only input returns uniform distribution
                            uniform = 1.0 / len(self.labels)
                            all_probs.append({lbl: round(uniform, 4) for lbl in self.labels})
                        else:
                            all_probs.append({
                                lbl: round(float(probs[idx][j]), 4)
                                for j, lbl in enumerate(self.labels)
                            })
        else:
            # Baseline scikit-learn pipeline
            for i in range(0, len(sanitized), batch_size):
                batch_texts = sanitized[i : i + batch_size]
                proc_batch = [t if t else " " for t in batch_texts]
                probs_arr = self.baseline_model.predict_proba(proc_batch)
                for idx, orig_t in enumerate(batch_texts):
                    if not orig_t:
                        uniform = 1.0 / len(self.labels)
                        all_probs.append({lbl: round(uniform, 4) for lbl in self.labels})
                    else:
                        all_probs.append({
                            lbl: round(float(probs_arr[idx][j]), 4)
                            for j, lbl in enumerate(self.labels)
                        })

        return all_probs

    def predict(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Predict sentiment label with confidence score and uncertainty flag."""
        is_single = isinstance(texts, str)
        raw_list = [texts] if is_single else list(texts)
        probs_list = self.predict_proba(raw_list, batch_size=batch_size)

        results: List[Dict[str, Any]] = []
        for orig_text, probs in zip(raw_list, probs_list):
            sanitized = self._sanitize_input(orig_text)
            
            if not sanitized:
                results.append({
                    "text": orig_text,
                    "label": "uncertain",
                    "confidence": 0.0,
                    "is_uncertain": True,
                    "probabilities": probs,
                    "model": self.model_name,
                })
                continue

            best_label = max(probs, key=probs.get)
            confidence = probs[best_label]

            # Uncertainty rule: if top class probability < threshold, mark as uncertain
            is_uncertain = confidence < self.threshold
            final_label = "uncertain" if is_uncertain else best_label

            results.append({
                "text": orig_text,
                "label": final_label,
                "predicted_class": best_label,
                "confidence": confidence,
                "is_uncertain": is_uncertain,
                "probabilities": probs,
                "model": self.model_name,
            })

        return results[0] if is_single else results
