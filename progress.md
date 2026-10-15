# Project Progress & Milestones: Telugu Text Classifier

This document tracks all completed engineering work, verified benchmarks, and technical milestones achieved across each phase of the project.

---

## 📅 Phase 1: Foundation, Data Engineering & Classical Baseline (Completed)

### 1. Data Engineering & Leak-Free Pipeline
- **Dataset**: `mounikaiiith/Telugu_Sentiment` (35,142 raw native Telugu sentences from IIIT Hyderabad).
- **Cleansing**: Unicode NFC normalization, whitespace stripping, and zero-width non-joiner standardization.
- **Leakage Prevention**: Built near-duplicate detection using character 5-gram shingling with Jaccard similarity threshold $\ge 0.8$. Removed 143 exact duplicates and 538 near-duplicates **before** splitting, preventing near-duplicate leakage between train and evaluation sets.
- **Stratified Partitioning**: 70% Train (24,123), 15% Validation (5,169), 15% Test (5,169) saved to `data/processed/` with seed 42.
- **Comprehensive Documentation**: Generated `reports/data_report.md` detailing token distributions, class imbalance (53.8% neutral, 27.1% pos, 19.1% neg), and vocabulary statistics.

### 2. Classical Baseline & Evaluation Framework
- **Model Tuning**: Evaluated character n-gram (2–5) TF-IDF pipelines with Logistic Regression. Tuned inverse regularization parameter $C$ exclusively on validation split.
- **Baseline Test Performance**:
  - **Macro-F1**: 0.6483 (95% Bootstrap CI: 0.6346 – 0.6618)
  - **Accuracy**: 0.6607
  - **Weighted-F1**: 0.6626
- **Test Set Isolation**: Ensured the test split remained completely untouched during tuning, evaluated once via `scripts/evaluate_baseline.py`.
- **Test Suite**: Built comprehensive pytest tests for data cleaning, deduplication, leak-free splits, and metrics.

---

## 📅 Phase 2: Transformer Fine-Tuning & Statistical Rigor (Completed)

### 1. Multi-Model, Multi-Seed Transformer Training
- Fine-tuned 3 transformer architectures on Google Colab GPU across 3 fixed random seeds ($42, 7, 123$) for 3 epochs with AdamW, linear warmup, and fp16:
  - `ai4bharat/IndicBERTv2-MLM-only` (12-layer Indic language model, ~278M params)
  - `google/muril-base-cased` (Multilingual Representations for Indian Languages)
  - `xlm-roberta-base` (General multilingual transformer)
- Managed Colab resource constraints, checkpoints, and model weights safely saved to Google Drive.

### 2. Verified Benchmark Evaluation (5,169 Held-Out Test Samples)
- **Official Results**:
  | Model | Seed | Test Macro-F1 | 95% Bootstrap CI | Test Accuracy | Mean Val Macro-F1 |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **IndicBERTv2-MLM-only** | **123** | **0.7332** | **[0.7197, 0.7459]** | **0.7361** | **0.7264** |
  | **MuRIL** | 7 | 0.7054 | [0.6921, 0.7171] | 0.7048 | 0.7060 |
  | **XLM-RoBERTa** | 42 | 0.6837 | [0.6708, 0.6964] | 0.6821 | 0.6807 |
  | **TF-IDF + LogReg** | 42 | 0.6483 | [0.6346, 0.6618] | 0.6607 | 0.6520 |

### 3. Statistical Significance Testing
- Conducted 1,000-iteration paired bootstrap hypothesis test:
  - IndicBERTv2 vs. Baseline delta: **+0.0849 (+8.49% Macro-F1)**
  - $p < 0.001$, confirmed with non-overlapping 95% confidence intervals ($[0.7197, 0.7459]$ vs $[0.6346, 0.6618]$).

---

## 📅 Phase 3: Error Analysis, Production Inference & Deployment (Completed)

### 1. Qualitative & Quantitative Error Analysis
- Extracted misclassifications and manually inspected 100 test errors.
- Established a quantitative 5-category taxonomy (`reports/error_analysis.md`, `results/100_errors_labeled.csv`):
  1. **Subtle Neutral / Contextual Boundary (41%)**
  2. **Mixed Sentiment / Contrasting Clauses (27%)**
  3. **Colloquial & Code-Switching (17%)**
  4. **Implicit Negation (9%)**
  5. **Sarcasm & Rhetorical Irony (6%)**

### 2. Production Inference Engine (`src/predict.py`)
- Created `TeluguSentimentPredictor` supporting:
  - Batching and single-string inference
  - Class probability distributions
  - **Uncertainty confidence thresholding** ($\ge 0.50$ flags ~13.5% low-confidence predictions as `uncertain` for human-in-the-loop review)
  - Resilient input sanitization (handling empty strings, whitespace, non-strings, long texts)
  - Automatic fallback between fine-tuned transformer and serialized baseline pipeline.
- Verified with 7 passing tests in `tests/test_predict.py`.

### 3. Interactive Web Application (`app.py`)
- Developed Gradio application featuring:
  - Real-time dual-model comparison (IndicBERTv2 vs. Baseline)
  - Softmax probability distribution bar charts
  - Uncertainty threshold slider with immediate visual flag
  - 6 curated Telugu test examples spanning positive, negative, neutral news, mixed sentiment, and code-mixing
  - Integrated benchmark disclosures and error analysis limitations.

### 4. Hugging Face Model Card & Space Packaging
- Wrote official `MODEL_CARD.md` with complete metadata, verified metrics, hyperparameters, usage code snippets, and bias disclosures.
- Created standalone Space deployment directory in `deployment/hf_space/` (`app.py`, `README.md` with YAML header, `requirements.txt`).
- Created token-safe `scripts/push_to_hub.py` supporting dry-run validation and uploading models and Spaces via `HF_TOKEN`.

### 5. Final Documentation & Portfolio Assets
- Updated `README.md` with full benchmark tables, architecture diagrams, and reproduction commands.
- Authored `docs/resume_bullets.md` featuring 3 high-impact resume bullets and a LinkedIn post draft.