from __future__ import annotations

import numpy as np
import pytest
import torch

from predict import EfficientNetBinaryClassifier, inference_for_image, model_available


@pytest.fixture
def dummy_model():
    model = EfficientNetBinaryClassifier(pretrained=False)
    model.eval()
    return model


def test_model_available_false_when_missing():
    assert model_available("missing_checkpoint.pth") is False


def test_inference_output_structure(dummy_model):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    result = inference_for_image(dummy_model, image, device=torch.device("cpu"))
    assert set(result.keys()) >= {"class_name", "display_label", "probability", "raw_score", "threshold", "status"}
    assert result["status"] == "ok"
    assert 0 <= result["probability"] <= 1
