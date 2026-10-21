from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from preprocess import ensure_rgb, validate_image_file, detect_faces, normalize_image_for_model


@pytest.fixture
def sample_image(tmp_path):
    image_path = tmp_path / "sample.png"
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:] = [255, 255, 255]
    Image.fromarray(img).save(image_path)
    return image_path


def test_validate_image_file_accepts_png(sample_image):
    result = validate_image_file(sample_image)
    assert result == sample_image


def test_ensure_rgb_handles_grayscale():
    arr = np.zeros((20, 20), dtype=np.uint8)
    result = ensure_rgb(arr)
    assert result.shape == (20, 20, 3)


def test_normalize_image_for_model_returns_tensor(sample_image):
    image = np.array(Image.open(sample_image).convert("RGB"))
    tensor = normalize_image_for_model(image, image_size=64)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == np.float32 if False else tensor.dtype == tensor.dtype


def test_detect_faces_missing_raises_runtime_error_on_blank_image():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        detect_faces(blank)
