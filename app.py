"""Gradio Web Application for Telugu Sentiment Classification.

Provides interactive sentiment inference, dual-model comparison (IndicBERTv2
vs. TF-IDF baseline), probability distributions, uncertainty thresholding,
and scientific benchmarks with error analysis disclosures.
"""

from pathlib import Path
from typing import Dict, Tuple
import gradio as gr

from src.predict import TeluguSentimentPredictor

PROJECT_ROOT = Path(__file__).resolve().parent

# Initialize primary predictor and baseline predictor
primary_predictor = TeluguSentimentPredictor(threshold=0.50)
baseline_predictor = TeluguSentimentPredictor(
    model_path=PROJECT_ROOT / "results/baseline_model.joblib",
    threshold=0.50,
)

LABEL_EMOJIS = {
    "pos": "🟢 Positive (సానుకూలం)",
    "positive": "🟢 Positive (సానుకూలం)",
    "neg": "🔴 Negative (ప్రతికూలం)",
    "negative": "🔴 Negative (ప్రతికూలం)",
    "neutral": "⚪ Neutral (తటస్థం)",
    "neu": "⚪ Neutral (తటస్థం)",
    "uncertain": "⚠️ Uncertain (సందిగ్ధత - Low Confidence)",
}


def analyze_sentiment(
    text: str, confidence_threshold: float
) -> Tuple[Dict[str, float], str, Dict[str, float], str, str]:
    """Run dual-model sentiment inference with uncertainty thresholding."""
    if not text or not text.strip():
        empty_probs = {"positive": 0.3333, "neutral": 0.3334, "negative": 0.3333}
        empty_badge = "⚠️ **Input is empty**: Please enter Telugu text for sentiment evaluation."
        return empty_probs, empty_badge, empty_probs, empty_badge, "No input provided."

    # Update threshold dynamically
    primary_predictor.threshold = confidence_threshold
    baseline_predictor.threshold = confidence_threshold

    # Primary Model Prediction
    primary_res = primary_predictor.predict(text)
    p_probs = primary_res["probabilities"]
    p_label = primary_res["label"]
    p_conf = primary_res["confidence"]
    p_uncertain = primary_res["is_uncertain"]

    # Baseline Model Prediction
    base_res = baseline_predictor.predict(text)
    b_probs = base_res["probabilities"]
    b_label = base_res["label"]
    b_conf = base_res["confidence"]
    b_uncertain = base_res["is_uncertain"]

    # Format Primary Status
    if p_uncertain:
        primary_status = (
            f"### {LABEL_EMOJIS['uncertain']}\n"
            f"**Top Class:** `{primary_res['predicted_class']}` ({p_conf * 100:.1f}%) < "
            f"Threshold (`{confidence_threshold * 100:.0f}%`). Marked as uncertain."
        )
    else:
        primary_status = (
            f"### {LABEL_EMOJIS.get(p_label, p_label)}\n"
            f"**Confidence:** `{p_conf * 100:.1f}%` | **Model:** `{primary_res['model']}`"
        )

    # Format Baseline Status
    if b_uncertain:
        baseline_status = (
            f"### {LABEL_EMOJIS['uncertain']}\n"
            f"**Top Class:** `{base_res['predicted_class']}` ({b_conf * 100:.1f}%) < "
            f"Threshold (`{confidence_threshold * 100:.0f}%`). Marked as uncertain."
        )
    else:
        baseline_status = (
            f"### {LABEL_EMOJIS.get(b_label, b_label)}\n"
            f"**Confidence:** `{b_conf * 100:.1f}%` | **Model:** `{base_res['model']}`"
        )

    # Comparison summary
    p_top = primary_res["predicted_class"]
    b_top = base_res["predicted_class"]
    if p_top == b_top:
        comparison_md = (
            f"✅ **Models Agree**: Both models identified `{p_top}` as the primary sentiment. "
            f"(IndicBERTv2: `{p_conf * 100:.1f}%`, Baseline: `{b_conf * 100:.1f}%`)"
        )
    else:
        comparison_md = (
            f"⚡ **Model Disagreement**: Primary predicted `{p_top}` (`{p_conf * 100:.1f}%`), "
            f"while Baseline predicted `{b_top}` (`{b_conf * 100:.1f}%`). "
            f"IndicBERTv2 captures subword morphology and contextual nuances that n-gram TF-IDF misses."
        )

    return p_probs, primary_status, b_probs, baseline_status, comparison_md


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


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks demo."""
    with gr.Blocks(title="Telugu Sentiment Classifier | తెలుగు సెంటిమెంట్ వర్గీకరణ") as demo:
        with gr.Column():
            gr.HTML(
                """
                <div class="header-box">
                    <h1>Telugu Sentiment Classifier</h1>
                    <p>తెలుగు భాషా సెంటిమెంట్ విశ్లేషణ | IndicBERTv2 Fine-Tuned Transformer vs. Baseline</p>
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
                        info="Predictions with top confidence below this threshold are flagged as 'Uncertain' for human-in-the-loop review.",
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
                fn=lambda: ("", 0.50, {}, "Awaiting input...", {}, "Awaiting input...", "Enter text and click Analyze to view model comparison."),
                inputs=[],
                outputs=[text_input, confidence_slider, primary_label_plot, primary_status_box, baseline_label_plot, baseline_status_box, comparison_output],
            )

            with gr.Accordion("📈 Benchmark Performance & Statistical Verification (Phase 2 Results)", open=False):
                gr.Markdown(
                    r"""
| Model Architecture | Test Macro-F1 | 95% Bootstrap CI | Test Accuracy | Mean Val Macro-F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IndicBERTv2-MLM-only** | **0.7332** | **[0.7197, 0.7459]** | **0.7361** | **0.7264** | **Best Model (+8.49% over baseline)** |
| **MuRIL (google/muril-base-cased)** | 0.7054 | [0.6921, 0.7171] | 0.7048 | 0.7060 | Competitive (+5.71%) |
| **XLM-RoBERTa (xlm-roberta-base)** | 0.6837 | [0.6708, 0.6964] | 0.6821 | 0.6807 | Baseline multilingual (+3.54%) |
| **TF-IDF + Logistic Regression** | 0.6483 | [0.6346, 0.6618] | 0.6607 | 0.6520 | Classical Baseline |

> **Statistical Significance**: Paired bootstrap hypothesis test (1,000 resamples): $\Delta = +0.0849$, $p < 0.001$. The 95% confidence intervals are completely non-overlapping, confirming statistically rigorous improvement.
                    """
                )

            with gr.Accordion("⚠️ Error Analysis & Known Limitations", open=False):
                gr.Markdown(
                    r"""
Based on qualitative error analysis of 100 manual test error samples:
- **Subtle Neutral Boundaries (41%)**: Statements containing factual reporting with subtle positive or negative context (e.g. government announcements, administrative updates).
- **Mixed Sentiment / Contrasting Clauses (27%)**: Sentences containing positive and negative aspects joined by conjunctions like "కానీ" (but) or "అయితే" (however).
- **Colloquial & Code-Switching (17%)**: Telugu written in Latin script (Telugu script with English loan words or transliteration).
- **Implicit Negation (9%)**: Negation expressed through contextual tone without explicit negative particles (లేదు, కాదు).
- **Sarcasm & Rhetorical Irony (6%)**: Sarcastic praise or rhetoric where literal tokens convey positive sentiment contrary to intended meaning.

**Production Recommendation**: For high-stakes applications, combine confidence thresholding ($\ge 0.50$) with human-in-the-loop review for uncertain predictions.
                    """
                )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
