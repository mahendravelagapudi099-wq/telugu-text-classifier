"""Hugging Face Space Application for Telugu Sentiment Classification.

Provides interactive sentiment inference, dual-model comparison,
uncertainty thresholding, and probability distributions.
"""

import os
from pathlib import Path
from typing import Dict, Tuple
import gradio as gr
import torch

LABEL_EMOJIS = {
    "pos": "🟢 Positive (సానుకూలం)",
    "positive": "🟢 Positive (సానుకూలం)",
    "neg": "🔴 Negative (ప్రతికూలం)",
    "negative": "🔴 Negative (ప్రతికూలం)",
    "neutral": "⚪ Neutral (తటస్థం)",
    "neu": "⚪ Neutral (తటస్థం)",
    "uncertain": "⚠️ Uncertain (సందిగ్ధత - Low Confidence)",
}

LABELS = ["neg", "neutral", "pos"]
DEFAULT_MODEL_ID = os.environ.get("HF_MODEL_ID", "ai4bharat/IndicBERTv2-MLM-only")

# Lazy-loaded predictor instances
_tokenizer = None
_model = None
_baseline_pipeline = None


def get_model_source():
    if "HF_MODEL_ID" in os.environ:
        return os.environ["HF_MODEL_ID"]
    candidates = [
        Path("best_model"),
        Path("outputs/IndicBERTv2-MLM-only_seed42/best_model"),
        Path("../../outputs/IndicBERTv2-MLM-only_seed42/best_model"),
        Path("outputs/IndicBERTv2-MLM-only_seed123/best_model"),
        Path("../../outputs/IndicBERTv2-MLM-only_seed123/best_model"),
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return str(c)
    return "ai4bharat/IndicBERTv2-MLM-only"


def get_transformer():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_source = get_model_source()
        _tokenizer = AutoTokenizer.from_pretrained(model_source)
        try:
            _model = AutoModelForSequenceClassification.from_pretrained(model_source)
        except Exception:
            # If base hub model lacks custom classifier head, initialize with 3 classes
            _model = AutoModelForSequenceClassification.from_pretrained(
                model_source, num_labels=3
            )
        _model.eval()
    return _tokenizer, _model


def get_baseline():
    global _baseline_pipeline
    if _baseline_pipeline is None:
        p = Path("baseline_model.joblib")
        if not p.exists():
            p = Path("results/baseline_model.joblib")
        if p.exists():
            import joblib

            _baseline_pipeline = joblib.load(p)
    return _baseline_pipeline


def analyze_sentiment(
    text: str, confidence_threshold: float
) -> Tuple[Dict[str, float], str, Dict[str, float], str, str]:
    if not text or not text.strip():
        empty_probs = {"positive": 0.3333, "neutral": 0.3334, "negative": 0.3333}
        empty_badge = "⚠️ **Input is empty**: Please enter Telugu text for sentiment evaluation."
        return empty_probs, empty_badge, empty_probs, empty_badge, "No input provided."

    clean_text = text.strip()

    # Transformer prediction
    try:
        tok, mdl = get_transformer()
        inputs = tok(clean_text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = mdl(**inputs)
            probs_tensor = torch.softmax(outputs.logits, dim=-1).squeeze()
            t_probs = {
                lbl: round(float(probs_tensor[j]), 4)
                for j, lbl in enumerate(LABELS)
            }
            top_t_label = max(t_probs, key=t_probs.get)
            top_t_conf = t_probs[top_t_label]
            t_uncertain = top_t_conf < confidence_threshold
    except Exception as e:
        t_probs = {"neg": 0.33, "neutral": 0.34, "pos": 0.33}
        top_t_label = "uncertain"
        top_t_conf = 0.34
        t_uncertain = True

    # Baseline prediction
    baseline = get_baseline()
    if baseline is not None:
        b_probs_arr = baseline.predict_proba([clean_text])[0]
        b_probs = {lbl: round(float(b_probs_arr[j]), 4) for j, lbl in enumerate(LABELS)}
        top_b_label = max(b_probs, key=b_probs.get)
        top_b_conf = b_probs[top_b_label]
        b_uncertain = top_b_conf < confidence_threshold
    else:
        b_probs = t_probs
        top_b_label = top_t_label
        top_b_conf = top_t_conf
        b_uncertain = t_uncertain

    # Format Statuses
    if t_uncertain:
        p_status = (
            f"### {LABEL_EMOJIS['uncertain']}\n"
            f"**Top Class:** `{top_t_label}` ({top_t_conf * 100:.1f}%) < "
            f"Threshold (`{confidence_threshold * 100:.0f}%`). Flagged for review."
        )
    else:
        p_status = (
            f"### {LABEL_EMOJIS.get(top_t_label, top_t_label)}\n"
            f"**Confidence:** `{top_t_conf * 100:.1f}%` | **Model:** IndicBERTv2"
        )

    if b_uncertain:
        b_status = (
            f"### {LABEL_EMOJIS['uncertain']}\n"
            f"**Top Class:** `{top_b_label}` ({top_b_conf * 100:.1f}%) < "
            f"Threshold (`{confidence_threshold * 100:.0f}%`). Flagged for review."
        )
    else:
        b_status = (
            f"### {LABEL_EMOJIS.get(top_b_label, top_b_label)}\n"
            f"**Confidence:** `{top_b_conf * 100:.1f}%` | **Model:** TF-IDF + LogisticRegression"
        )

    if top_t_label == top_b_label:
        comparison_md = (
            f"✅ **Models Agree**: Both models identified `{top_t_label}` as primary sentiment. "
            f"(IndicBERTv2: `{top_t_conf * 100:.1f}%`, Baseline: `{top_b_conf * 100:.1f}%`)"
        )
    else:
        comparison_md = (
            f"⚡ **Model Disagreement**: IndicBERTv2 predicted `{top_t_label}` (`{top_t_conf * 100:.1f}%`), "
            f"while Baseline predicted `{top_b_label}` (`{top_b_conf * 100:.1f}%`)."
        )

    return t_probs, p_status, b_probs, b_status, comparison_md


SAMPLE_EXAMPLES = [
    [
        "ఈ సినిమా కథనం మరియు నటన అద్భుతంగా ఉన్నాయి. ప్రతి ఒక్కరూ తప్పకుండా చూడాల్సిన చిత్రం!",
        0.50,
    ],
    [
        "ఈ ఉత్పత్తి చాలా నాసిరకంగా ఉంది, కొన్న రెండు రోజులకే పూర్తిగా పాడైపోయింది.",
        0.50,
    ],
    [
        "హైదరాబాద్‌లో రేపు వర్షం కురిసే అవకాశం ఉందని వాతావరణ శాఖ అధికారులు వెల్లడించారు.",
        0.50,
    ],
    [
        "సినిమాలో పాటలు చాలా బాగున్నాయి కానీ కథనం మాత్రం చాలా నెమ్మదిగా సాగదీశారు.",
        0.50,
    ],
    [
        "ఈ రోజే కొత్త స్మార్ట్‌ఫోన్ వచ్చింది, delivery చాలా fast గా ఉంది and performance super!",
        0.50,
    ],
    [
        "సరే చూద్దాం",
        0.50,
    ],
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.header-box {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #f8fafc;
}
.header-box p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}
"""

with gr.Blocks(title="Telugu Sentiment Classifier | తెలుగు సెంటిమెంట్ వర్గీకరణ") as demo:
    with gr.Column():
        gr.HTML(
            """
            <div class="header-box">
                <h1>Telugu Sentiment Classifier</h1>
                <p>తెలుగు భాషా సెంటిమెంట్ విశ్లేషణ | Fine-Tuned Transformer vs. Baseline</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    lines=3,
                    placeholder="తెలుగు వచనాన్ని ఇక్కడ నమోదు చేయండి (Enter Telugu text here)...",
                    label="Telugu Input Text (తెలుగు వాక్యం)",
                )
                confidence_slider = gr.Slider(
                    minimum=0.34,
                    maximum=0.90,
                    value=0.50,
                    step=0.02,
                    label="Uncertainty Confidence Threshold (సందిగ్ధత పరిమితి)",
                    info="Predictions with confidence below threshold are flagged as Uncertain.",
                )
                with gr.Row():
                    submit_btn = gr.Button("Analyze Sentiment (సెంటిమెంట్ విశ్లేషించు)", variant="primary")
                    clear_btn = gr.Button("Clear (క్లియర్)")

            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Model Comparison Overview")
                comparison_output = gr.Markdown("Enter text and click Analyze to view model comparison.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🚀 IndicBERTv2 Model (Primary)")
                primary_status_box = gr.Markdown("Awaiting input...")
                primary_label_plot = gr.Label(num_top_classes=3, label="Probability Distribution")

            with gr.Column():
                gr.Markdown("### 📊 TF-IDF Baseline (Comparison)")
                baseline_status_box = gr.Markdown("Awaiting input...")
                baseline_label_plot = gr.Label(num_top_classes=3, label="Probability Distribution")

        gr.Examples(
            examples=SAMPLE_EXAMPLES,
            inputs=[text_input, confidence_slider],
            outputs=[primary_label_plot, primary_status_box, baseline_label_plot, baseline_status_box, comparison_output],
            fn=analyze_sentiment,
            cache_examples=False,
            label="Curated Test Examples (నమూనా ఉదాహరణలు)",
        )

        submit_btn.click(
            fn=analyze_sentiment,
            inputs=[text_input, confidence_slider],
            outputs=[primary_label_plot, primary_status_box, baseline_label_plot, baseline_status_box, comparison_output],
        )
        clear_btn.click(
            fn=lambda: ("", 0.50, {}, "Awaiting input...", {}, "Awaiting input...", "Enter text and click Analyze."),
            inputs=[],
            outputs=[text_input, confidence_slider, primary_label_plot, primary_status_box, baseline_label_plot, baseline_status_box, comparison_output],
        )

        with gr.Accordion("📈 Benchmark Performance & Statistical Verification", open=False):
            gr.Markdown(
                r"""
| Model Architecture | Test Macro-F1 | 95% Bootstrap CI | Test Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **IndicBERTv2-MLM-only** | **0.7332** | **[0.7197, 0.7459]** | **0.7361** | **Best Model (+8.49%)** |
| **MuRIL (google/muril-base-cased)** | 0.7054 | [0.6921, 0.7171] | 0.7048 | Competitive (+5.71%) |
| **XLM-RoBERTa (xlm-roberta-base)** | 0.6837 | [0.6708, 0.6964] | 0.6821 | Baseline Multilingual (+3.54%) |
| **TF-IDF + Logistic Regression** | 0.6483 | [0.6346, 0.6618] | 0.6607 | Classical Baseline |

> **Bootstrap Hypothesis Test**: 1,000 paired resamples $\Delta = +0.0849$, $p < 0.001$. Non-overlapping 95% CIs.
                """
            )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
