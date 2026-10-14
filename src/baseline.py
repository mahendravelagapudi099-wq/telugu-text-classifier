"""TF-IDF + Logistic Regression baselines with validation-only tuning.

Two pipeline variants: word n-grams and char_wb n-grams. The tuning grid
is scored on the validation set only; the test set is never touched here.
"""

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# (variant, ngram_range) pairs defining the pipeline families
VARIANTS: list[tuple[str, tuple[int, int]]] = [
    ("word", (1, 1)),
    ("word", (1, 2)),
    ("char", (2, 4)),
    ("char", (2, 5)),
]
MIN_DF_GRID = [1, 5]
C_GRID = [0.5, 1.0, 4.0]


def build_pipeline(
    variant: str,
    ngram_range: tuple[int, int],
    min_df: int,
    C: float,
    seed: int,
) -> Pipeline:
    """Build a TF-IDF + LogisticRegression pipeline for the given variant."""
    if variant not in {"word", "char"}:
        raise ValueError(f"unknown variant: {variant!r}")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word" if variant == "word" else "char_wb",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def tune_baselines(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Fit every grid combination on train, score macro-F1 on validation.

    Returns {"best": params+score, "all_results": every combination,
    "best_model": the fitted pipeline}. The test set must not be passed.
    """
    all_results: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best: dict[str, Any] | None = None

    for variant, ngram_range in VARIANTS:
        for min_df in MIN_DF_GRID:
            for C in C_GRID:
                pipe = build_pipeline(variant, ngram_range, min_df, C, seed)
                pipe.fit(train_texts, train_labels)
                preds = pipe.predict(val_texts)
                macro_f1 = float(f1_score(val_labels, preds, average="macro"))
                row = {
                    "variant": variant,
                    "ngram_range": list(ngram_range),
                    "min_df": min_df,
                    "C": C,
                    "val_macro_f1": macro_f1,
                }
                all_results.append(row)
                if (
                    best is None
                    or macro_f1 > best["val_macro_f1"]
                    or (
                        macro_f1 == best["val_macro_f1"]
                        and (variant, ngram_range, min_df, C)
                        < (best["variant"], tuple(best["ngram_range"]), best["min_df"], best["C"])
                    )
                ):
                    best = row
                    best_model = pipe

    all_results.sort(
        key=lambda r: (
            -r["val_macro_f1"],
            r["variant"],
            tuple(r["ngram_range"]),
            r["min_df"],
            r["C"],
        )
    )
    return {"best": best, "all_results": all_results, "best_model": best_model}
