from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str | None = None):
        self.model = model
        self.target_layer = target_layer or "features"
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _resolve_layer(self) -> nn.Module | None:
        target = self.model
        if hasattr(target, "model"):
            target = target.model
        attr_path = self.target_layer.split(".") if isinstance(self.target_layer, str) else []
        for part in attr_path:
            if part.endswith("]"):
                base, index = part[:-1].split("[")
                if hasattr(target, base):
                    target = getattr(target, base)
                else:
                    return None
                if isinstance(target, (list, tuple)):
                    try:
                        target = target[int(index)]
                    except (ValueError, IndexError):
                        return None
                continue
            if not hasattr(target, part):
                return None
            target = getattr(target, part)
        if isinstance(target, nn.Module):
            return target
        if isinstance(target, (list, tuple)) and target:
            return target[-1]
        return None

    def _register_hooks(self) -> None:
        layer = self._resolve_layer()
        if layer is None:
            layer = getattr(self.model, "model", self.model)
            if hasattr(layer, "features"):
                layer = layer.features[-1]
            else:
                raise RuntimeError("The requested target layer was not found in the model.")

        def forward_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_inputs: Any, grad_outputs: Any) -> None:
            self.gradients = grad_outputs[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_tensor.dim() != 4:
            raise ValueError("Grad-CAM requires a batch of 4D tensors [N, C, H, W].")
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        target = logits[:, target_class]
        target.sum().backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed to capture activations or gradients.")
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam[0, 0].cpu().numpy()
        cam_np = cv2.resize(cam_np, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam_np


def generate_gradcam_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = image * (1.0 - alpha) + heatmap_color * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_gradcam_visualizations(original: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, output_dir: str | Path, prefix: str = "gradcam") -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_path / f"{prefix}_heatmap.png"
    overlay_path = output_path / f"{prefix}_overlay.png"
    original_path = output_path / f"{prefix}_original.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"original": str(original_path), "heatmap": str(heatmap_path), "overlay": str(overlay_path)}
