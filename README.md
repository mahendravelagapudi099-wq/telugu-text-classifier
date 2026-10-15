# Telugu Text Classifier (తెలుగు సెంటిమెంట్ వర్గీకరణ)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IndicBERTv2-orange)](MODEL_CARD.md)
[![Gradio Demo](https://img.shields.io/badge/UI-Gradio%20Demo-red)](app.py)

An end-to-end, leak-free Telugu sentiment classification system comparing classical TF-IDF baselines against modern multilingual and Indic transformers (`IndicBERTv2`, `MuRIL`, `XLM-RoBERTa`) across 3 random seeds, complete with uncertainty thresholding, error taxonomy, and an interactive Gradio web application.

---

## 📌 Problem & Scope

3-class sentiment classification of native Telugu sentences:
- `pos` (Positive / సానుకూలం)
- `neutral` (Neutral / తటస్థం)
- `neg` (Negative / ప్రతికూలం)

The dataset exhibits natural real-world class imbalance (53.8% neutral, 27.1% positive, 19.1% negative). Therefore, **Macro-F1 is our primary evaluation metric** — weighting each class equally and preventing majority neutral classes from inflating the overall score.

---

## 📊 Dataset & Leakage Prevention

We use the [mounikaiiith/Telugu_Sentiment](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment) dataset containing 35,142 native Telugu sentences across news, films, cricket, and politics, curated by IIIT Hyderabad (Marreddy et al., 2022 / CC-BY-4.0).

### Strict Leak-Free Preprocessing:
1. **Unicode NFC Normalization & Whitespace Cleansing**: Standardized Telugu vowel signs and zero-width characters.
2. **Pre-Split Near-Duplicate Deduplication**: Removed 143 exact duplicates and 538 near duplicates using character 5-gram shingling with Jaccard similarity threshold $\ge 0.8$ **before** splitting. This ensures zero near-duplicates span train and test sets.
3. **Stratified 70 / 15 / 15 Partitioning**: Yields fixed subsets saved to `data/processed/` with seed 42:
   - **Train**: 24,123 sentences
   - **Validation**: 5,169 sentences
   - **Test**: 5,169 sentences (strictly touched only during final unbiased evaluation)

Full data statistics and vocabulary report: [reports/data_report.md](reports/data_report.md).

---

## 🏆 Benchmark Results

All models were evaluated on the held-out test split of **5,169 examples** across random seeds ($42, 7, 123$).

| Model Architecture | Base Pretrained Checkpoint | Test Macro-F1 | 95% Bootstrap CI | Test Accuracy | Mean Val Macro-F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IndicBERTv2** | `ai4bharat/IndicBERTv2-MLM-only` | **0.7332** | **[0.7197, 0.7459]** | **0.7361** | **0.7264** | **Best Model (+8.49% over baseline)** |
| **MuRIL** | `google/muril-base-cased` | 0.7054 | [0.6921, 0.7171] | 0.7048 | 0.7060 | Competitive (+5.71%) |
| **XLM-RoBERTa** | `xlm-roberta-base` | 0.6837 | [0.6708, 0.6964] | 0.6821 | 0.6807 | Baseline Multilingual (+3.54%) |
| **TF-IDF Baseline** | Character n-grams (2–5) + LogReg | 0.6483 | [0.6346, 0.6618] | 0.6607 | 0.6520 | Classical Baseline |

### Statistical Significance Verification
- **Paired Bootstrap Hypothesis Test (1,000 resamples)**: IndicBERTv2 vs. TF-IDF Baseline: $\Delta = +0.0849$ ($+8.49\%$ Macro-F1 gain, $p < 0.001$).
- **Non-Overlapping Confidence Intervals**: The 95% bootstrap confidence interval for IndicBERTv2 ($[0.7197, 0.7459]$) does not overlap with the TF-IDF baseline ($[0.6346, 0.6618]$), mathematically confirming that the transformer's superior performance is statistically significant.

### 📈 Visualizations & Training Convergence

| Test Set Confusion Matrix | IndicBERTv2 Training & Validation Loss Curves |
| :---: | :---: |
| ![Confusion Matrix](results/confusion_matrix_test.png) | ![IndicBERTv2 Loss Curves](results/loss_curves_indicbert.png) |
| *Confusion Matrix on 5,169 Test Samples* | *Loss convergence over 3 fine-tuning epochs* |

---

## 🔍 Error Analysis & Taxonomy

Rather than treating the remaining 26.7% error rate as opaque, we manually inspected and categorized 100 misclassified test examples ([reports/error_analysis.md](reports/error_analysis.md)):

```
Error Category Distribution (100 Samples):
├── 41% Subtle Neutral / Contextual Boundary (Factual reporting with subtle sentiment tone)
├── 27% Mixed Sentiment / Contrasting Clauses ("Songs were great, but story dragged")
├── 17% Colloquial & Code-Switching (Telugu in Latin script / English loan words)
├──  9% Implicit Negation (Contextual negation without explicit "లేదు/కాదు" particles)
└──  6% Sarcasm & Rhetorical Irony (Surface positive words with negative intent)
```

Detailed qualitative error report with Telugu examples: [reports/error_analysis.md](reports/error_analysis.md) and labeled CSV: [results/100_errors_labeled.csv](results/100_errors_labeled.csv).

---

## 🚀 Quickstart & Inference

### 1. Installation
```bash
git clone https://github.com/mahendravelagapudi099-wq/telugu-text-classifier.git
cd Telugu-Text-Classifier
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

### 2. Python Inference API
```python
from src.predict import TeluguSentimentPredictor

# Initialize predictor with uncertainty thresholding (default 0.50)
predictor = TeluguSentimentPredictor(threshold=0.50)

# Single prediction
result = predictor.predict("ఈ సినిమా చాలా బాగుంది!")
print(result)
# Output:
# {
#   'text': 'ఈ సినిమా చాలా బాగుంది!',
#   'label': 'pos',
#   'confidence': 0.975,
#   'is_uncertain': False,
#   'probabilities': {'neg': 0.012, 'neutral': 0.013, 'pos': 0.975}
# }

# Batch prediction
batch_results = predictor.predict([
    "ఈ ఉత్పత్తి చాలా బాగుంది!",
    "నాకు ఏమీ అర్థం కాలేదు",
    "ఈ రోజే కొత్త ఫోన్ వచ్చింది, delivery fast గా ఉంది"
])
```

### 3. Launch Interactive Gradio Demo
```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to explore the dual-model comparison interface with real test examples, probability distributions, and the uncertainty slider.

---

## 📦 Hugging Face Deployment

- **Model Card**: See [MODEL_CARD.md](MODEL_CARD.md) for full architecture parameters, dataset documentation, and ethical limitations.
- **Space Deployment Package**: Self-contained files located in `deployment/hf_space/`.
- **Automated Safe Upload Script**:
  ```bash
  # Dry-run validation (no token required)
  python scripts/push_to_hub.py --dry-run --space-repo username/telugu-sentiment-demo

  # Live upload using HF_TOKEN environment variable
  $env:HF_TOKEN="hf_your_token_here"
  python scripts/push_to_hub.py --space-repo username/telugu-sentiment-demo
  ```

---

## 🧪 Testing & Verification

Run the complete test suite (data cleaning, deduplication, leak-free splits, metrics, and inference):
```bash
pytest tests/
```

---

## 📁 Repository Structure

```
├── configs/
│   └── config.yaml               # Reproducibility seeds, split ratios, dedup parameters
├── src/
│   ├── data.py                   # Normalization, shingle deduplication, stratified split
│   ├── baseline.py               # TF-IDF + Logistic Regression pipelines & grid search
│   ├── evaluate.py               # F1, accuracy, bootstrap CIs, confusion matrix plotting
│   ├── predict.py                # Production inference engine with uncertainty thresholding
│   └── utils.py                  # YAML configuration loaders
├── scripts/
│   ├── prepare_data.py           # Pipeline: raw -> processed data
│   ├── analyze_data.py           # Vocabulary & class distribution analysis
│   ├── train_baseline.py         # Classical baseline training & model serialization
│   ├── evaluate_baseline.py      # Final baseline evaluation on test split
│   ├── train_transformer.py      # Multi-seed Hugging Face trainer pipeline
│   ├── eval_transformer.py       # Transformer test evaluation & prediction export
│   ├── compare_models.py         # Paired bootstrap significance testing
│   ├── generate_error_analysis.py# Samples and prepares 100 test errors
│   └── push_to_hub.py            # Token-safe Hugging Face Hub / Space deployment
├── deployment/
│   └── hf_space/                 # Standalone Gradio app for Hugging Face Spaces
├── reports/
│   ├── data_report.md            # Comprehensive dataset documentation
│   └── error_analysis.md         # 5-category error taxonomy with Telugu examples
├── results/
│   ├── 100_errors_labeled.csv    # 100 manually reviewed test errors
│   ├── baseline_model.joblib     # Serialized TF-IDF + LogReg pipeline
│   └── confusion_matrix_test.png # Confusion matrix on test set
├── tests/                        # 8 test modules verifying end-to-end functionality
├── app.py                        # Gradio interactive web application
├── MODEL_CARD.md                 # Official Hugging Face Model Card
├── progress.md                   # Chronological project milestones and progress log
└── docs/
    └── resume_bullets.md         # Portfolio resume bullets & LinkedIn announcement
```

---

## 📜 Citation & Credits
- **Dataset**: Marreddy et al., *"Clickbait Detection and Sentiment Analysis in Telugu"*, TALLIP 2022.
- **Model**: AI4Bharat, *"IndicBERT v2: Towards High-Quality Pre-trained Language Models for Indic Languages"*, arXiv:2312.05445.
