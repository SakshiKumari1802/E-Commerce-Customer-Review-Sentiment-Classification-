import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from sentiment.config import LABEL_NAMES


def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    output_dir: Path,
) -> dict:
    """Save metrics, a classification report, and a confusion-matrix image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(LABEL_NAMES)
    target_names = [LABEL_NAMES[label] for label in labels]
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": target_names,
    }
    safe_name = model_name.lower().replace(" ", "_")
    (output_dir / f"{safe_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axis,
    )
    axis.set(title=f"{model_name} confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / f"{safe_name}_confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def save_results_table(metrics_list: list[dict], output_dir: Path) -> pd.DataFrame:
    """Write a compact comparison table without overwriting detailed reports."""
    columns = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    table = pd.DataFrame([{key: item[key] for key in columns} for item in metrics_list])
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "model_comparison.csv", index=False)
    return table
