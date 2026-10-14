"""Tokenize the 5 sample Telugu sentences with each candidate model.

Reports token counts per sentence to compare tokenization efficiency.
Downloads tokenizer files only (no model weights).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

MODELS = [
    "ai4bharat/IndicBERTv2-MLM-only",
    "google/muril-base-cased",
    "xlm-roberta-base",
]


def main() -> None:
    samples = json.loads(
        Path("results/_tokenizer_samples.json").read_text(encoding="utf-8")
    )
    out = {}
    for model_id in MODELS:
        tok = AutoTokenizer.from_pretrained(model_id)
        counts = [len(tok.encode(s)) for s in samples]
        out[model_id] = {
            "tokenizer_class": tok.__class__.__name__,
            "vocab_size": tok.vocab_size,
            "token_counts": counts,
            "mean_tokens": sum(counts) / len(counts),
        }
        print(f"{model_id} [{tok.__class__.__name__}] vocab={tok.vocab_size}")
        for s, c in zip(samples, counts):
            print(f"  {c:3d} tokens | {s[:60]}")
    Path("results/tokenizer_check.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
