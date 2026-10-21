from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from config import DEFAULT_CHECKPOINT, DEFAULT_TRAINING_CONFIG, DATASET_ROOT, OUTPUTS_DIR, ensure_directories, get_device
from preprocess import set_seed, validate_dataset_root


class ImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], image_size: int = 224):
        self.items = items
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_dataset(dataset_dir: str | Path) -> list[tuple[str, int]]:
    folders = validate_dataset_root(dataset_dir)
    items: list[tuple[str, int]] = []
    for label_name, files in folders.items():
        target_label = 0 if label_name == "REAL" else 1
        for file_path in files:
            items.append((str(file_path), target_label))
    return items


def split_dataset(items: list[tuple[str, int]], valid_split: float = 0.15, test_split: float = 0.15, seed: int = 42) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    if valid_split + test_split >= 1.0:
        raise ValueError("Validation and test splits must sum to less than 1.0")
    random.Random(seed).shuffle(items)
    total = len(items)
    test_count = int(total * test_split)
    valid_count = int(total * valid_split)
    train_count = total - test_count - valid_count
    train_items = items[:train_count]
    test_items = items[train_count:train_count + test_count]
    valid_items = items[train_count + test_count:]
    return train_items, valid_items, test_items


def evaluate_predictions(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def create_model(pretrained: bool = True, device: str = "cpu") -> nn.Module:
    from predict import EfficientNetBinaryClassifier
    model = EfficientNetBinaryClassifier(pretrained=pretrained)
    return model.to(device)


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    ensure_directories()
    set_seed(args.random_seed)
    device = torch.device(args.device)
    items = build_dataset(args.dataset_dir)
    train_items, valid_items, test_items = split_dataset(items, args.valid_split, args.test_split, seed=args.random_seed)

    train_loader = DataLoader(ImageDataset(train_items, image_size=args.image_size), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(ImageDataset(valid_items, image_size=args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(ImageDataset(test_items, image_size=args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = create_model(pretrained=args.use_pretrained, device=str(device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = None
    best_metric = -1.0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        y_true, y_pred, y_score = [], [], []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            y_true.extend(labels.cpu().numpy().astype(int).ravel().tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).ravel().tolist())
            y_score.extend(probs.detach().cpu().numpy().ravel().tolist())

        train_metrics = evaluate_predictions(y_true, y_pred, y_score)
        val_loss = 0.0
        val_true, val_pred, val_score = [], [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                val_true.extend(labels.cpu().numpy().astype(int).ravel().tolist())
                val_pred.extend((probs >= 0.5).float().cpu().numpy().astype(int).ravel().tolist())
                val_score.extend(probs.detach().cpu().numpy().ravel().tolist())
        val_metrics = evaluate_predictions(val_true, val_pred, val_score)
        current_metric = float(val_metrics["f1"])
        scheduler.step(current_metric)
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_items),
            "val_loss": val_loss / len(valid_items),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        })
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={history[-1]['train_loss']:.4f}, val_f1={history[-1]['val_f1']:.4f}, val_auc={history[-1]['val_roc_auc']}")

        if current_metric > best_metric + args.min_delta:
            best_metric = current_metric
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": DEFAULT_TRAINING_CONFIG.as_dict(),
        "class_map": {"REAL": 0, "FAKE": 1},
        "model_architecture": "EfficientNet-B0",
        "training_metadata": {
            "epochs_completed": len(history),
            "best_val_f1": best_metric,
            "dataset_dir": str(args.dataset_dir),
            "history": history,
            "device": str(device),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Best model saved to {checkpoint_path}")

    test_true, test_pred, test_score = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            test_true.extend(labels.cpu().numpy().astype(int).ravel().tolist())
            test_pred.extend((probs >= 0.5).float().cpu().numpy().astype(int).ravel().tolist())
            test_score.extend(probs.detach().cpu().numpy().ravel().tolist())

    test_metrics = evaluate_predictions(test_true, test_pred, test_score)
    metrics_path = OUTPUTS_DIR / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2)
    return {"checkpoint": str(checkpoint_path), "history": history, "test_metrics": test_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Explainable Deepfake Detection model.")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_ROOT), help="Path to dataset root containing real/ and fake/ directories.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--valid-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--use-pretrained", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
