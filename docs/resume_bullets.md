# Telugu Text Sentiment Classifier — Resume Bullets & LinkedIn Post

---

## 📄 Resume Bullets

### Bullet 1 (Model Performance & Benchmarking)
- **Fine-tuned and evaluated multi-seed IndicBERTv2, MuRIL, and XLM-RoBERTa models** against a tuned character n-gram TF-IDF baseline on a 34,460-row Telugu sentiment corpus; achieved state-of-the-art **0.7332 Test Macro-F1** (an **+8.49% absolute improvement**, $p < 0.001$ via 1,000 paired bootstrap resamples, 95% CI: 0.7197–0.7459).

### Bullet 2 (Data Engineering & Leakage Prevention)
- **Engineered an end-to-end reproducible NLP pipeline** incorporating Unicode NFC normalization and character 5-gram Jaccard deduplication ($\ge 0.8$) before stratified 70/15/15 splitting, eliminating cross-split train/test contamination and ensuring zero data leakage across 5,169 held-out test evaluations.

### Bullet 3 (Error Taxonomy & Production Guardrails)
- **Constructed a quantitative 5-category error taxonomy** over 100 manually reviewed model failure cases (identifying subtle neutral boundaries at 41%, mixed sentiment clauses at 27%, and colloquial code-switching at 17%); developed an inference engine featuring **uncertainty confidence thresholding** ($\ge 0.50$) and deployed a dual-model comparison Gradio application to Hugging Face Spaces.

---

## 💼 LinkedIn Post Draft

**Headline:** Outperforming n-gram baselines in low-resource Indian NLP: Fine-Tuning IndicBERTv2 on Telugu Sentiment Classification 🇮🇳📊

Telugu is spoken by over 95 million people worldwide, yet building robust NLP systems for Dravidian languages presents distinct challenges: rich agglutinative morphology, complex sandhi compounds, and widespread code-mixing with English.

Over the past few weeks, I built and evaluated an end-to-end, leakage-free text classification pipeline comparing classical models and multilingual transformers on the 34,460-sample Telugu Sentiment dataset (IIIT Hyderabad / CC-BY-4.0).

Here are key engineering takeaways and verified results:

1️⃣ **Strict Leakage Prevention Before Splitting:**
Standard random train/test splits often leak near-duplicate text into the evaluation set. By applying character 5-gram shingling with Jaccard similarity ($\ge 0.8$) **prior** to stratified 70/15/15 splitting, cross-split leakage was completely prevented across 5,169 test samples.

2️⃣ **Statistical Rigor Across 3 Random Seeds:**
Rather than trusting a single lucky run, I benchmarked `ai4bharat/IndicBERTv2-MLM-only`, `google/muril-base-cased`, and `xlm-roberta-base` across 3 seeds (42, 7, 123):
- **IndicBERTv2-MLM-only**: **0.7332 Macro-F1** (95% CI: 0.7197–0.7459, Accuracy: 73.61%)
- **MuRIL**: 0.7054 Macro-F1 (95% CI: 0.6921–0.7171)
- **XLM-RoBERTa**: 0.6837 Macro-F1 (95% CI: 0.6708–0.6964)
- **Tuned TF-IDF + Logistic Regression Baseline**: 0.6483 Macro-F1 (95% CI: 0.6346–0.6618)

A paired bootstrap hypothesis test (1,000 resamples) demonstrated a statistically significant delta of **+8.49% Macro-F1** ($p < 0.001$) with non-overlapping confidence intervals.

3️⃣ **Why 73.3% and Not 95%? Quantitative Error Analysis:**
High accuracy numbers without error analysis are meaningless in production. I manually labeled and categorized 100 test errors to find why the model stumbled:
- 41% Subtle Neutral / Contextual boundaries (factual news with mild sentiment flavor)
- 27% Mixed Sentiment / Contrasting clauses ("songs were great, but story dragged")
- 17% Colloquial & Code-Switching (Telugu written in Latin script / English loan words)
- 9% Implicit Negation without explicit negative particles
- 6% Sarcasm & Rhetorical Irony

4️⃣ **Production Guardrails:**
To handle borderline and out-of-distribution inputs safely, I implemented an uncertainty thresholding module ($\ge 0.50$ confidence) that automatically flags ambiguous inputs for human-in-the-loop review (~13.5% of cases), rather than blindly outputting low-confidence guesses.

Interactive Gradio demo & full open-source repo:
🔗 GitHub: https://github.com/mahendravelagapudi099-wq/telugu-text-classifier
🤗 Hugging Face Model & Space: [Link]

Special thanks to AI4Bharat and IIIT Hyderabad for open datasets and language models powering Indian language NLP!

#NLP #MachineLearning #Telugu #IndicNLP #DeepLearning #HuggingFace #Transformers #Python #AI
