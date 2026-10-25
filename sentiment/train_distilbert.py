import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sentiment.config import ARTIFACTS_DIR, DATA_DIR, LABEL_NAMES, RANDOM_SEED
from sentiment.evaluation import evaluate_predictions, save_results_table

MODEL_NAME = "distilbert-base-uncased"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one training update on eight training rows; skip test evaluation and model saving.",
    )
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    train_frame = pd.read_csv(DATA_DIR / "train.csv")
    validation_frame = pd.read_csv(DATA_DIR / "validation.csv")
    if args.smoke_test:
        train_frame = train_frame.head(8).copy()
        validation_frame = validation_frame.head(4).copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        dataset = dataset.rename_column("label", "labels")
        return dataset.map(tokenize, batched=True, remove_columns=["text", "sample_id"])

    train_data = to_dataset(train_frame)
    validation_data = to_dataset(validation_frame)
    set_seed(RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=LABEL_NAMES,
        label2id={name: label for label, name in LABEL_NAMES.items()},
    )
    use_cpu = not torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=(
            str(Path(tempfile.gettempdir()) / "ecommerce-distilbert-smoke")
            if args.smoke_test
            else str(ARTIFACTS_DIR / "distilbert_checkpoints")
        ),
        num_train_epochs=args.epochs,
        max_steps=args.smoke_steps if args.smoke_test else -1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="no" if args.smoke_test else "epoch",
        save_strategy="no" if args.smoke_test else "epoch",
        load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        use_cpu=use_cpu,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        logging_steps=1 if args.smoke_test else 100,
        disable_tqdm=args.smoke_test,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    if args.smoke_test:
        if not np.isfinite(train_result.training_loss):
            raise RuntimeError("DistilBERT smoke test produced a non-finite loss")
        print(
            f"Smoke test passed: {args.smoke_steps} training update(s) completed; "
            f"training_loss={train_result.training_loss:.6f}; test split untouched."
        )
        return

    test_frame = pd.read_csv(DATA_DIR / "test.csv")
    test_data = to_dataset(test_frame)
    predictions = trainer.predict(test_data)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    metrics_dir = ARTIFACTS_DIR / "metrics"
    result = evaluate_predictions(
        "DistilBERT",
        test_frame["label"],
        predicted_labels,
        metrics_dir,
    )
    trainer.save_model(str(ARTIFACTS_DIR / "models" / "distilbert"))
    tokenizer.save_pretrained(str(ARTIFACTS_DIR / "models" / "distilbert"))

    existing = []
    for metric_file in metrics_dir.glob("*_metrics.json"):
        existing.append(json.loads(metric_file.read_text(encoding="utf-8")))
    save_results_table(existing, metrics_dir)
    print(f"Finished held-out evaluation: DistilBERT (CPU={use_cpu})")
    print(result)


if __name__ == "__main__":
    main()
