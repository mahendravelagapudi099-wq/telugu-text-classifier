# Telugu Text Classifier

Classifies Telugu text sentiment and compares a classical baseline against
fine-tuned Indic transformers on identical, leakage-free splits.

**Phase 1 (current state):** dataset preparation, TF-IDF + Logistic
Regression baseline, and the evaluation framework. Transformer
fine-tuning is Phase 2; Hugging Face Spaces demo is Phase 3.

## Problem

3-class sentiment classification of native Telugu sentences:
`neg` (negative), `neutral`, `pos` (positive). The class distribution is
imbalanced (53.8% neutral / 27.1% pos / 19.1% neg), so **macro-F1 is the
primary metric** — unlike accuracy or weighted-F1, it weights every class
equally and does not let the majority class hide poor minority-class
performance.

## Dataset

[mounikaiiith/Telugu_Sentiment](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment)
— 35,142 native Telugu sentences (news, cricket, films, politics),
manually annotated by IIIT Hyderabad researchers (Marreddy et al.,
"TALLIP" 2022 / arXiv:2205.01204). License: **CC-BY-4.0**. The text is
native Telugu, not machine-translated.

Data preparation pools the official splits, cleans (Unicode NFC,
whitespace), removes 143 exact and 538 near duplicates (char-5-gram
Jaccard >= 0.8, removed **before** splitting so no near duplicate spans
two splits), then creates fixed stratified 70/15/15 train/validation/test
splits (seed 42) saved to `data/processed/`. Final: 34,461 rows
(24,123 / 5,169 / 5,169). Every model — including future transformers —
uses exactly these files; the test set is touched only by
`scripts/evaluate_baseline.py`.

Details and samples: [reports/data_report.md](reports/data_report.md).

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# .venv/bin/pip install -r requirements.txt          # Linux/macOS
```

## Reproduce

```bash
.venv/Scripts/python scripts/prepare_data.py     # download, clean, dedup, split -> data/processed/
.venv/Scripts/python scripts/analyze_data.py     # -> reports/data_report.md
.venv/Scripts/python scripts/train_baseline.py   # tune on train, select on validation -> results/baseline_model.joblib
.venv/Scripts/python scripts/evaluate_baseline.py  # final validation+test metrics, CI, CM, errors
.venv/Scripts/python -m pytest tests/            # run the test suite
```

## Results

| Model | Split | Macro-F1 | 95% CI | Accuracy | Weighted-F1 |
|---|---|---|---|---|---|
| TF-IDF char (2-5) + LogReg, C=4 | validation | 0.6520 | 0.6377–0.6655 | 0.6667 | 0.6685 |
| TF-IDF char (2-5) + LogReg, C=4 | test | 0.6483 | 0.6346–0.6618 | 0.6607 | 0.6626 |
| *(transformer — Phase 2)* | — | — | — | — | — |

Per-class test metrics and the confusion matrix:
`results/baseline_metrics.json`, `results/confusion_matrix_test.png`.
Misclassified test examples for manual categorization:
`results/errors_test.csv` (fill in the `my_error_category` column).

## Project layout

```
configs/config.yaml     # dataset, labels, seed, split ratios, dedup params, paths
src/data.py             # cleaning, dedup, stratified splitting
src/baseline.py         # TF-IDF + LogReg pipelines, validation-only tuning
src/evaluate.py         # metrics, bootstrap CI, CM plot, error export
src/utils.py            # config loading
scripts/                # prepare_data, analyze_data, train_baseline, evaluate_baseline
tests/                  # pytest suite (cleaning, dedup/leakage, splits, metrics)
data/                   # gitignored; regenerate via scripts/prepare_data.py
results/, reports/      # metrics, model artifacts, data report
```
