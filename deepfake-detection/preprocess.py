from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from config import DEFAULT_TRAINING_CONFIG, OUTPUTS_DIR


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _safe_file_size(path: str | os.PathLike[str]) -> int:
    return Path(path).stat().st_size


def validate_image_file(path: str | os.PathLike[str], max_size_mb: float = 20.0) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format for {file_path}")
    size_mb = _safe_file_size(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image exceeds size limit: {size_mb:.2f} MB > {max_size_mb} MB")
    try:
        with Image.open(file_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Corrupted or unreadable image: {file_path}") from exc
    return file_path


def validate_video_file(path: str | os.PathLike[str], max_duration_seconds: float = 180.0) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Video not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format for {file_path}")
    size_mb = _safe_file_size(file_path) / (1024 * 1024)
    if size_mb > 500:
        raise ValueError(f"Video file is too large: {size_mb:.2f} MB > 500 MB")
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to decode video: {file_path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / fps if fps and fps > 0 else 0.0
        if duration > max_duration_seconds:
            raise ValueError(f"Video is too long: {duration:.1f}s > {max_duration_seconds}s")
        return file_path
    finally:
        cap.release()


def load_image(path: str | os.PathLike[str]) -> np.ndarray:
    validated = validate_image_file(path)
    with Image.open(validated) as image:
        rgb = image.convert("RGB")
        return np.array(rgb)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.astype(np.uint8)


def normalize_image_for_model(image: np.ndarray, image_size: int = 224) -> torch.Tensor:
    image = ensure_rgb(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (tensor - mean.squeeze(0)) / std.squeeze(0)
    return normalized


def build_test_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if classifier.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded.")
    faces = classifier.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [tuple(map(int, box)) for box in faces]


def crop_face(image: np.ndarray, box: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    x, y, w, h = box
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + w + padding)
    y1 = min(image.shape[0], y + h + padding)
    return image[y0:y1, x0:x1]


def validate_dataset_root(dataset_dir: str | os.PathLike[str]) -> dict[str, list[Path]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    real_dir = root / "real"
    fake_dir = root / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError("Expected dataset/real and dataset/fake directories.")

    real_files = sorted(p for p in real_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    fake_files = sorted(p for p in fake_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)

    if not real_files:
        raise ValueError(f"No real images found in {real_dir}")
    if not fake_files:
        raise ValueError(f"No fake images found in {fake_dir}")

    for label_dir, files in (("REAL", real_files), ("FAKE", fake_files)):
        for file_path in files:
            try:
                validate_image_file(file_path)
            except ValueError as exc:
                raise ValueError(f"Invalid image in {label_dir}: {file_path}") from exc
    return {"REAL": real_files, "FAKE": fake_files}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_file(path: str | os.PathLike[str]) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_outputs_dir(subdir: str = "") -> Path:
    base = OUTPUTS_DIR / subdir if subdir else OUTPUTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_image_shape(image: np.ndarray, min_dim: int = 32) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.size == 0:
        raise ValueError("Input image is empty")
    arr = ensure_rgb(image)
    if arr.shape[0] < min_dim or arr.shape[1] < min_dim:
        raise ValueError(f"Image too small: {arr.shape[:2]}")
    return arr
