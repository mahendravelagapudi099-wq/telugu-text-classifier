"""Fine-tune a transformer for Telugu sentiment classification.

Uses the EXACT Phase 1 splits (data/processed/{train,validation}.csv);
the test set is never touched here. Selects the best checkpoint by
validation macro-F1 with early stopping; supports class-weighted loss,
fp16 on GPU, and gradient accumulation.

Example:
    python scripts/train_transformer.py --model google/muril-base-cased \
        --seed 42 --output-dir outputs/muril_seed42
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils import load_config  # noqa: E402


def compute_class_weights(labels: list[str], counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency class weights: n_total / (n_classes * n_class)."""
    n_total = sum(counts.values())
    weights = [n_total / (len(labels) * counts[label]) for label in labels]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """Trainer with a class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def plot_loss_curves(log_history: list[dict], out_path: Path) -> None:
    """Save train loss and validation loss/macro-F1 curves as PNG."""
    train = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    evals = [(e["epoch"], e.get("eval_loss"), e.get("eval_macro_f1")) for e in log_history if "eval_loss" in e]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(*zip(*train), label="train loss")
    eval_epochs, eval_losses, _ = zip(*[(e, l, f) for e, l, f in evals if l is not None])
    ax1.plot(eval_epochs, eval_losses, "o-", label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    f1_points = [(e, f) for e, _, f in evals if f is not None]
    if f1_points:
        ax2.plot(*zip(*f1_points), "o-", color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation macro-F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    cfg = load_config()["training"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--lr", type=float, default=cfg["learning_rate"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--max-length", type=int, default=cfg["max_length"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=cfg["gradient_accumulation_steps"]
    )
    parser.add_argument("--train-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--val-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    labels: list[str] = cfg["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    processed = Path(cfg["paths"]["processed"])

    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "validation.csv")
    if args.train_limit:
        train_df = train_df.head(args.train_limit)
    if args.val_limit:
        val_df = val_df.head(args.val_limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        ds = ds.map(tokenize, batched=True)
        ds = ds.map(lambda b: {"labels": [label2id[l] for l in b["label"]]}, batched=True)
        return ds.remove_columns(["label"])

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label={i: l for l, i in label2id.items()}, label2id=label2id
    )

    class_weights = None
    if not args.no_class_weights:
        counts = train_df["label"].value_counts().to_dict()
        class_weights = compute_class_weights(labels, counts)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5 removed warmup_ratio; derive integer steps instead.
    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.gradient_accumulation_steps))
    warmup_steps = int(cfg["training"]["warmup_ratio"] * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=warmup_steps,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to=[],
    )

    def compute_metrics(eval_pred) -> dict:
        logits, targets = eval_pred
        preds = logits.argmax(-1)
        return {"macro_f1": f1_score(targets, preds, average="macro", zero_division=0)}

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["early_stopping_patience"])],
    )

    start = time.time()
    trainer.train()
    training_seconds = time.time() - start

    best_val_f1 = max(
        (e.get("eval_macro_f1", 0.0) for e in trainer.state.log_history if "eval_macro_f1" in e),
        default=0.0,
    )
    plot_loss_curves(trainer.state.log_history, output_dir / "loss_curves.png")
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    summary = {
        "model": args.model,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "class_weights": class_weights is not None,
        "best_val_macro_f1": float(best_val_f1),
        "training_seconds": round(training_seconds, 1),
        "gpu": gpu,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "log_history": trainer.state.log_history,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
