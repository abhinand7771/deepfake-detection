from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from config import DEFAULT_CHECKPOINT, DEFAULT_TRAINING_CONFIG, detect_cuda_status, get_device
from gradcam import GradCAM, generate_gradcam_overlay
from preprocess import detect_faces, crop_face, ensure_image_shape, load_image, validate_image_file


MODEL_METADATA_KEYS = {"class_map", "config", "model_architecture", "training_metadata"}


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except Exception:  # pragma: no cover
                weights = None
        model = models.efficientnet_b0(weights=weights)
        if weights is not None:
            in_features = model.classifier[1].in_features
        else:
            in_features = 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model_from_checkpoint(checkpoint_path: str | os.PathLike[str] | None = None, device: torch.device | str | None = None) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else Path(DEFAULT_CHECKPOINT)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("config", DEFAULT_TRAINING_CONFIG.as_dict())
        meta = checkpoint.get("training_metadata", {})
        if "class_map" in checkpoint:
            class_map = checkpoint["class_map"]
        else:
            class_map = {"REAL": 0, "FAKE": 1}
        model = EfficientNetBinaryClassifier(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model.to(device or get_device()), {"class_map": class_map, "config": model_config, "training_metadata": meta}
    raise ValueError("Checkpoint does not contain a valid model state.")


def model_available(checkpoint_path: str | os.PathLike[str] | None = None) -> bool:
    target = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    return target.exists()


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def inference_for_image(model: nn.Module, image: np.ndarray, device: torch.device, threshold: float = 0.5, inconclusive_threshold: float = 0.1, image_size: int = 224) -> dict[str, Any]:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    image = ensure_image_shape(image)
    tensor = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(-1)
    score = float(probs.item())
    raw_score = float(logits.squeeze(-1).item())
    prob_real = float(1.0 - score)
    prob_fake = float(score)
    if abs(score - 0.5) < inconclusive_threshold:
        label = "Inconclusive"
        display = "Inconclusive"
    elif score >= threshold:
        label = "FAKE"
        display = "Likely Manipulated"
    else:
        label = "REAL"
        display = "Likely Real"
    return {
        "class_name": label,
        "display_label": display,
        "probability": score,
        "raw_score": raw_score,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "threshold": threshold,
        "inconclusive_threshold": inconclusive_threshold,
        "device": str(device),
        "status": "ok",
    }


def predict_image_from_path(path: str | os.PathLike[str], checkpoint_path: str | os.PathLike[str] | None = None, threshold: float = 0.5, inconclusive_threshold: float = 0.1) -> dict[str, Any]:
    validated_path = validate_image_file(path)
    image = load_image(validated_path)
    model, meta = load_model_from_checkpoint(checkpoint_path)
    device = get_device()
    model = model.to(device)
    model.eval()
    faces = detect_faces(image)
    if not faces:
        return {
            "status": "no_face_detected",
            "message": "No face was detected in the image. The model was not asked to make a guess without a valid face crop.",
            "faces": [],
            "device": str(device),
            "available_model": True,
        }
    results = []
    for box in faces:
        crop = crop_face(image, box, padding=10)
        result = inference_for_image(model, crop, device=device, threshold=threshold, inconclusive_threshold=inconclusive_threshold)
        result["face_box"] = list(box)
        result["status"] = "ok"
        results.append(result)
    return {
        "status": "ok",
        "message": "Prediction completed for detected face crops.",
        "faces": results,
        "device": str(device),
        "available_model": True,
        "metadata": meta,
    }


def predict_image_upload(image_array: np.ndarray | None, checkpoint_path: str | os.PathLike[str] | None = None, threshold: float = 0.5, inconclusive_threshold: float = 0.1) -> dict[str, Any]:
    if image_array is None:
        raise ValueError("No image data provided.")
    image = ensure_image_shape(image_array)
    model, meta = load_model_from_checkpoint(checkpoint_path)
    device = get_device()
    model = model.to(device)
    model.eval()
    faces = detect_faces(image)
    if not faces:
        return {
            "status": "no_face_detected",
            "message": "No face was detected in the uploaded image.",
            "faces": [],
            "device": str(device),
            "available_model": True,
            "metadata": meta,
        }
    results = []
    for box in faces:
        crop = crop_face(image, box, padding=10)
        output = inference_for_image(model, crop, device=device, threshold=threshold, inconclusive_threshold=inconclusive_threshold)
        output["face_box"] = list(box)
        results.append(output)
    return {
        "status": "ok",
        "message": "Prediction completed for uploaded image.",
        "faces": results,
        "device": str(device),
        "available_model": True,
        "metadata": meta,
    }


if __name__ == "__main__":
    print("Prediction API loaded. Use predict_image_from_path() or predict_image_upload() by importing this module.")
