"""Model adapters used by the inference pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class SeverityPredictor:
    """
    Load a trained ResNet18 checkpoint and classify BGR pothole crops.

    Supports both v1 checkpoints (plain fc layer) and v3 checkpoints
    (Sequential dropout + linear head saved as ``fc.*`` after key remapping).
    Optionally blends model probabilities with a bounding-box area heuristic
    for improved calibration in video inference.
    """

    def __init__(
        self,
        model_path: Path,
        classes: tuple[str, ...],
        image_size: int,
        device: str | int,
        bbox_heuristic_weight: float = 0.35,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Severity checkpoint not found: {model_path}")

        self.model_path = model_path
        self.classes = classes
        self.bbox_heuristic_weight = bbox_heuristic_weight
        self.device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else device
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))

        try:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        # Try loading directly; if that fails, strip Sequential wrapper keys
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # v3 checkpoint may have been saved with Sequential head keys
            # (fc.0.* dropout, fc.1.* linear) — strip them down to fc.*
            remapped: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("fc.1."):
                    remapped[k.replace("fc.1.", "fc.")] = v
                elif k.startswith("fc.0."):
                    pass  # dropout has no weights
                else:
                    remapped[k] = v
            self.model.load_state_dict(remapped)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, crop_bgr) -> tuple[str, float, dict[str, float]]:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot classify an empty crop")

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()

        class_id = int(probabilities.argmax().item())
        scores = {
            name: float(probabilities[index].item())
            for index, name in enumerate(self.classes)
        }
        return self.classes[class_id], scores[self.classes[class_id]], scores

    def predict_with_bbox(
        self,
        crop_bgr,
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> tuple[str, float, dict[str, float]]:
        """
        Predict severity and blend with a bounding-box area heuristic.

        The heuristic encodes the real-world intuition that a pothole
        occupying a large fraction of the camera frame is more severe:

            relative_area < 0.015  → skewed toward Low
            0.015 – 0.07           → skewed toward Medium
            > 0.07                 → skewed toward High

        ``bbox_heuristic_weight`` (default 0.20) controls how much the
        bbox signal shifts the model's softmax output.
        """
        _, _, model_scores = self.predict(crop_bgr)
        blended = self.blend_bbox_heuristic(
            model_scores, bbox, frame_size, self.classes, self.bbox_heuristic_weight
        )
        best_class = max(blended, key=blended.get)
        return best_class, blended[best_class], blended

    @staticmethod
    def blend_bbox_heuristic(
        model_probs: dict[str, float],
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
        classes: tuple[str, ...],
        weight: float = 0.20,
    ) -> dict[str, float]:
        """
        Linearly blend model probabilities with a bbox-area prior.

        ``classes`` must be ordered  [low, medium, high].
        ``weight`` = 0 → pure model; 1 → pure heuristic.
        """
        x1, y1, x2, y2 = bbox
        fw, fh = frame_size
        if fw <= 0 or fh <= 0:
            return model_probs  # guard against degenerate frames

        area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / (fw * fh)

        # Soft heuristic priors  [low, medium, high]
        if area_frac < 0.015:
            bbox_p = [0.65, 0.25, 0.10]
        elif area_frac < 0.035:
            bbox_p = [0.35, 0.50, 0.15]
        elif area_frac < 0.07:
            bbox_p = [0.15, 0.55, 0.30]
        elif area_frac < 0.12:
            bbox_p = [0.08, 0.22, 0.70]
        else:
            # Massive pothole occupying > 12% of screen (severe road crater)
            bbox_p = [0.02, 0.10, 0.88]

        blended: dict[str, float] = {}
        for i, cls in enumerate(classes):
            blended[cls] = (1.0 - weight) * model_probs.get(cls, 0.0) + weight * bbox_p[i]

        # Re-normalise to sum to 1 (should already be ~1, but float safety)
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended
