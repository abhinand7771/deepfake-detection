from __future__ import annotations

import numpy as np
import torch

from gradcam import GradCAM, generate_gradcam_overlay
from predict import EfficientNetBinaryClassifier


def test_gradcam_output_dimensions_and_values():
    model = EfficientNetBinaryClassifier(pretrained=False)
    model.eval()
    tensor = torch.rand(1, 3, 224, 224)
    gradcam = GradCAM(model)
    heatmap = gradcam(tensor, target_class=1)
    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    overlay = generate_gradcam_overlay(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), heatmap)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
