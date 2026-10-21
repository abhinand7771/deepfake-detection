from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = MODELS_DIR / "best_model.pth"


@dataclass
class TrainingConfig:
    dataset_dir: str | Path = DATASET_ROOT
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 2
    random_seed: int = 42
    valid_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    use_pretrained: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    threshold: float = 0.5
    inconclusive_threshold: float = 0.1
    label_map: dict[str, int] = field(default_factory=lambda: {"REAL": 0, "FAKE": 1})
    classes: list[str] = field(default_factory=lambda: ["REAL", "FAKE"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "workers": self.workers,
            "random_seed": self.random_seed,
            "valid_split": self.valid_split,
            "test_split": self.test_split,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "use_pretrained": self.use_pretrained,
            "device": self.device,
            "threshold": self.threshold,
            "inconclusive_threshold": self.inconclusive_threshold,
            "label_map": self.label_map,
            "classes": self.classes,
        }


DEFAULT_TRAINING_CONFIG = TrainingConfig()


def ensure_directories() -> None:
    for path in (DATASET_ROOT, MODELS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_cuda_status() -> dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_model_status() -> dict[str, Any]:
    exists = DEFAULT_CHECKPOINT.exists()
    return {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "exists": exists,
        "available": exists,
        "architecture": "EfficientNet-B0",
    }
