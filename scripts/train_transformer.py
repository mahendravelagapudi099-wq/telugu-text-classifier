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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils import load_config


def compute_class_weights(labels: list[str], counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency class weights: n_total / (n_classes * n_class)."""
    n_total = sum(counts.values())
    weights = [n_total / (len(labels) * max(counts[label], 1)) for label in labels]
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
    if train:
        ax1.plot(*zip(*train), label="train loss")
    if evals:
        eval_epochs = [e[0] for e in evals if e[1] is not None]
        eval_losses = [e[1] for e in evals if e[1] is not None]
        if eval_epochs:
            ax1.plot(eval_epochs, eval_losses, "o-", label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    f1_points = [(e[0], e[2]) for e in evals if e[2] is not None]
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
    parser.add_argument("--output-dir", type=str, default="outputs/run")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "validation.csv")
    if args.train_limit:
        train_df = train_df.head(args.train_limit)
    if args.val_limit:
        val_df = val_df.head(args.val_limit)

    label2id = {lbl: i for i, lbl in enumerate(cfg["labels"])}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )

    # Integer labels are set in pandas BEFORE building the Dataset: a
    # datasets map that overwrites an existing string column casts the new
    # values back to the string dtype, which silently drops labels later.
    def to_dataset(df: pd.DataFrame) -> Dataset:
        prepared = df[["text", "label"]].copy()
        prepared["labels"] = prepared["label"].map(label2id)
        ds = Dataset.from_pandas(prepared[["text", "labels"]], preserve_index=False)
        return ds.map(tokenize_fn, batched=True)

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(cfg["labels"]), id2label=id2label, label2id=label2id
    )

    train_labels_mapped = train_ds["labels"]
    counts = {i: train_labels_mapped.count(i) for i in range(len(cfg["labels"]))}
    class_weights = compute_class_weights(cfg["labels"], {cfg["labels"][k]: v for k, v in counts.items()})

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = logits.argmax(axis=-1)
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        accuracy = float((preds == labels).mean())
        return {"macro_f1": macro_f1, "accuracy": accuracy}

    # transformers 5 removed warmup_ratio; derive integer steps from the ratio.
    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["early_stopping_patience"])],
    )

    start_time = time.time()
    trainer.train()
    training_seconds = time.time() - start_time

    eval_results = trainer.evaluate()

    plot_loss_curves(trainer.state.log_history, output_dir / "loss_curves.png")

    summary = {
        "model": args.model,
        "seed": args.seed,
        "learning_rate": args.lr,
        "epochs_trained": float(trainer.state.epoch),
        "training_seconds": training_seconds,
        "best_val_macro_f1": eval_results.get("eval_macro_f1"),
        "best_val_accuracy": eval_results.get("eval_accuracy"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
