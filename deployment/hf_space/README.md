---
title: Telugu Sentiment Classifier
emoji: 🇮🇳
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Telugu text sentiment analysis using fine-tuned IndicBERTv2.
---

# Telugu Sentiment Classifier (తెలుగు సెంటిమెంట్ వర్గీకరణ)

Interactive Hugging Face Space for classifying Telugu sentences into Positive, Neutral, or Negative sentiment with dual-model comparison, probability distributions, and uncertainty thresholding.

## Features
- **Dual-Model Inference**: Compare fine-tuned IndicBERTv2 against the TF-IDF + Logistic Regression baseline.
- **Uncertainty Guardrails**: Dynamically flag low-confidence predictions ($\le \text{threshold}$) as `uncertain` to assist human-in-the-loop validation.
- **Visual Distributions**: Real-time bar charts of softmax probabilities across all three classes.
- **Pre-loaded Curated Test Examples**: Explore positive, negative, neutral news, mixed sentiment, and colloquial Telugu samples.
