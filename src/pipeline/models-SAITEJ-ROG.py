"""Model adapters used by the inference pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


class SeverityPredictor:
    """
    Load a trained ResNet18 checkpoint and classify BGR pothole crops.

    Supports both v1 checkpoints (plain fc layer) and v3 checkpoints
    (Sequential dropout + linear head saved as ``fc.*`` after key remapping).
    Uses batched Test-Time Augmentation (TTA) and blends model probabilities
    with a perspective-calibrated bounding-box area heuristic.
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
        self.image_size = image_size
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

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            remapped: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("fc.1."):
                    remapped[k.replace("fc.1.", "fc.")] = v
                elif k.startswith("fc.0."):
                    pass
                else:
                    remapped[k] = v
            self.model.load_state_dict(remapped)

        self.model.to(self.device)
        self.model.eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def _tta_batch(self, pil_img: Image.Image) -> torch.Tensor:
        """Build a 3-view TTA batch (full resize, horizontal flip, 90% center crop)."""
        sz = (self.image_size, self.image_size)
        v1 = self.normalize(TF.to_tensor(TF.resize(pil_img, sz)))
        v2 = self.normalize(TF.to_tensor(TF.hflip(TF.resize(pil_img, sz))))
        w, h = pil_img.size
        cw, ch = max(1, int(round(w * 0.90))), max(1, int(round(h * 0.90)))
        cropped = TF.center_crop(pil_img, [ch, cw])
        v3 = self.normalize(TF.to_tensor(TF.resize(cropped, sz)))
        return torch.stack([v1, v2, v3], dim=0).to(self.device)

    def predict(self, crop_bgr) -> tuple[str, float, dict[str, float]]:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot classify an empty crop")

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        batch = self._tta_batch(pil_img)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1).mean(dim=0).cpu()

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
        Predict severity with TTA and blend with a perspective-calibrated bbox heuristic.
        """
        _, _, model_scores = self.predict(crop_bgr)
        blended = self.blend_bbox_heuristic(
            model_scores, bbox, frame_size, self.classes, self.bbox_heuristic_weight
        )
        best_class = max(blended, key=blended.get)
        return best_class, blended[best_class], blended

    @staticmethod
    def _debias_ordinal_probs(
        raw_probs: dict[str, float],
        classes: tuple[str, ...],
    ) -> dict[str, float]:
        """Correct for v3 focal-oversampling bimodal split between low and high."""
        if len(classes) != 3:
            return raw_probs
        c_low, c_med, c_high = classes
        p_l = max(0.0, raw_probs.get(c_low, 0.0))
        p_m = max(0.0, raw_probs.get(c_med, 0.0))
        p_h = max(0.0, raw_probs.get(c_high, 0.0))

        # When both low and high are activated nearly equally, the ordinal expectation
        # E[y] = p_m + 2*p_h sits near 1.0 (Medium). Transfer bimodal ambiguity mass to Medium.
        bimodal_mass = 0.42 * min(p_l, p_h)
        adj_l = max(0.01, p_l - 0.45 * bimodal_mass)
        adj_m = p_m + bimodal_mass
        adj_h = max(0.01, p_h - 0.55 * bimodal_mass)
        total = adj_l + adj_m + adj_h
        return {
            c_low: adj_l / total,
            c_med: adj_m / total,
            c_high: adj_h / total,
        }

    @staticmethod
    def blend_bbox_heuristic(
        model_probs: dict[str, float],
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
        classes: tuple[str, ...],
        weight: float = 0.35,
    ) -> dict[str, float]:
        """
        Linearly blend ordinal-debiased model probabilities with a
        perspective-calibrated bbox-area prior.

        ``classes`` must be ordered [low, medium, high].
        ``weight`` = 0 → pure model; 1 → pure heuristic.
        """
        x1, y1, x2, y2 = bbox
        fw, fh = frame_size
        if fw <= 0 or fh <= 0:
            return model_probs

        cal_probs = SeverityPredictor._debias_ordinal_probs(model_probs, classes)
        area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / (fw * fh)
        y_center = (y1 + y2) / 2.0

        # Account for perspective foreshortening in forward-facing road scenes
        if fw > fh:
            y_norm = max(0.45, min(1.0, (y_center / fh - 0.15) / 0.72))
            effective_area = area_frac / (y_norm ** 1.15)
        else:
            effective_area = area_frac

        # Calibrated soft heuristic priors [low, medium, high]
        if effective_area < 0.005:
            bbox_p = [0.68, 0.24, 0.08]
        elif effective_area < 0.013:
            bbox_p = [0.22, 0.60, 0.18]
        elif effective_area < 0.022:
            bbox_p = [0.10, 0.32, 0.58]
        elif effective_area < 0.045:
            bbox_p = [0.06, 0.20, 0.74]
        else:
            bbox_p = [0.02, 0.10, 0.88]

        blended: dict[str, float] = {}
        for i, cls in enumerate(classes):
            blended[cls] = (1.0 - weight) * cal_probs.get(cls, 0.0) + weight * bbox_p[i]

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended
