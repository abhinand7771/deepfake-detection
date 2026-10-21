from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import DEFAULT_CHECKPOINT, OUTPUTS_DIR
from preprocess import validate_dataset_root
from train import ImageDataset, build_dataset, split_dataset


def evaluate_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, dataset_dir: str | Path = "dataset") -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
    dataset_root = Path(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    from predict import EfficientNetBinaryClassifier
    model = EfficientNetBinaryClassifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    items = build_dataset(dataset_root)
    _, _, test_items = split_dataset(items, valid_split=0.15, test_split=0.15, seed=42)
    loader = DataLoader(ImageDataset(test_items, image_size=224), batch_size=32, shuffle=False)

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            y_true.extend(labels.numpy().astype(int).tolist())
            y_pred.extend((probs[:, 1] >= 0.5).int().numpy().astype(int).tolist())
            y_score.extend(probs[:, 1].numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve.png")
    plt.close()

    return metrics


if __name__ == "__main__":
    evaluate_model()
