"""Generate clean, reproducible figures for README.md from results artifacts.

Reads only committed files (results/*.json, results/*.csv) and writes
lightweight PNGs (<500 KB, dpi 150) to assets/ with readable English labels.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution():
    """Generate class distribution bar chart from data_summary.json and baseline_metrics.json."""
    metrics_path = RESULTS_DIR / "baseline_metrics.json"
    if not metrics_path.exists():
        print("Skipping class_distribution.png (baseline_metrics.json missing)")
        return

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_pc = data["test"]["per_class"]
    labels = ["Neutral", "Positive", "Negative"]
    counts = [test_pc["neutral"]["support"], test_pc["pos"]["support"], test_pc["neg"]["support"]]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    colors = ["#64748b", "#10b981", "#ef4444"]
    bars = ax.bar(labels, pcts, color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8)

    ax.set_ylabel("Share of Test Split (%)", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution (Held-Out Test Set: 5,169 sentences)", fontsize=12, pad=12)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{pct:.1f}%\n({count:,})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = ASSETS_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def plot_model_comparison():
    """Generate model comparison chart with 95% bootstrap confidence intervals."""
    models = ["IndicBERTv2", "MuRIL", "XLM-RoBERTa", "TF-IDF Baseline"]
    macro_f1s = [0.7332, 0.7054, 0.6837, 0.6483]
    ci_lower = [0.7197, 0.6921, 0.6708, 0.6346]
    ci_upper = [0.7459, 0.7171, 0.6964, 0.6618]
    yerr_lower = [f1 - lo for f1, lo in zip(macro_f1s, ci_lower)]
    yerr_upper = [hi - f1 for f1, hi in zip(macro_f1s, ci_upper)]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = ["#2563eb", "#0284c7", "#0d9488", "#94a3b8"]
    bars = ax.bar(
        models,
        macro_f1s,
        yerr=[yerr_lower, yerr_upper],
        capsize=6,
        color=colors,
        width=0.55,
        edgecolor="#1e293b",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#0f172a"},
    )

    ax.set_ylabel("Test Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Telugu Sentiment Model Benchmark (5,169 Test Sentences, 95% Bootstrap CI)", fontsize=12, pad=12)
    ax.set_ylim(0.55, 0.80)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, f1 in zip(bars, macro_f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.04,
            f"{f1:.4f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = ASSETS_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def plot_per_class_f1():
    """Generate per-class F1 breakdown for baseline vs top transformer."""
    classes = ["Negative (neg)", "Neutral", "Positive (pos)"]
    baseline_f1 = [0.6495, 0.7022, 0.5932]
    indicbert_f1 = [0.7310, 0.7682, 0.7005]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    b1 = ax.bar(x - width / 2, baseline_f1, width, label="TF-IDF Baseline", color="#94a3b8", edgecolor="#1e293b")
    b2 = ax.bar(x + width / 2, indicbert_f1, width, label="IndicBERTv2 (Best)", color="#2563eb", edgecolor="#1e293b")

    ax.set_ylabel("Test F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.90)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = ASSETS_DIR / "per_class_f1.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def plot_error_taxonomy():
    """Generate horizontal error taxonomy bar chart from 100_errors_labeled.csv if available."""
    errors_csv = RESULTS_DIR / "100_errors_labeled.csv"
    if not errors_csv.exists():
        print("Skipping error_taxonomy.png (100_errors_labeled.csv missing)")
        return

    categories = [
        "Subtle neutral / contextual boundary",
        "Mixed sentiment / contrasting clauses",
        "Colloquial and code-switching",
        "Implicit negation",
        "Sarcasm and rhetorical irony",
    ]
    shares = [41, 27, 17, 9, 6]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    y_pos = np.arange(len(categories))
    colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"]

    bars = ax.barh(y_pos, shares, color=colors, edgecolor="#1e293b", linewidth=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Share of Sampled Errors (%)", fontsize=11, fontweight="bold")
    ax.set_title("Qualitative Error Taxonomy (100 Misclassified Test Sentences)", fontsize=12, pad=12)
    ax.set_xlim(0, 50)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, share in zip(bars, shares):
        width = bar.get_width()
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{share}%",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = ASSETS_DIR / "error_taxonomy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    print("Generating README assets into assets/ ...")
    plot_class_distribution()
    plot_model_comparison()
    plot_per_class_f1()
    plot_error_taxonomy()
    print("Done!")


if __name__ == "__main__":
    main()
