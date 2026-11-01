"""Multi-scale pyramid inference, tiled slicing, and confidence-weighted fusion."""

from __future__ import annotations

from typing import Any, Sequence

from src.detection.postprocess import (
    Detection,
    overlap_scores,
)


def fuse_multiscale_detections(
    scale_detections: Sequence[tuple[Detection, int]],
    *,
    fusion_iou: float = 0.40,
    consensus_boost: float = 0.22,
    num_scales: int = 1,
) -> list[Detection]:
    """Fuse detections from multiple scales with confidence-weighted boxes."""
    if not scale_detections:
        return []
    if num_scales <= 1:
        return [det for det, _ in scale_detections]

    ordered = sorted(scale_detections, key=lambda item: item[0].confidence, reverse=True)
    used = [False] * len(ordered)
    fused: list[Detection] = []

    for i, (anchor, _) in enumerate(ordered):
        if used[i]:
            continue
        used[i] = True
        cluster: list[tuple[Detection, int]] = [(anchor, ordered[i][1])]

        for j in range(i + 1, len(ordered)):
            if used[j]:
                continue
            candidate, scale = ordered[j]
            if candidate.class_id != anchor.class_id:
                continue
            iou, ios = overlap_scores(anchor, candidate)
            ax1, ay1, ax2, ay2 = anchor.xyxy
            bx1, by1, bx2, by2 = candidate.xyxy
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
            if iou >= fusion_iou or (ios >= 0.72 and area_ratio >= 0.35):
                cluster.append((candidate, scale))
                used[j] = True

        weights = [max(1e-4, det.confidence**2) for det, _ in cluster]
        weight_sum = sum(weights)
        fused_xyxy = tuple(
            int(round(sum(weight * det.xyxy[k] for weight, (det, _) in zip(weights, cluster)) / weight_sum))
            for k in range(4)
        )
        best_per_scale: dict[int, float] = {}
        for det, scale in cluster:
            best_per_scale[scale] = max(best_per_scale.get(scale, 0.0), det.confidence)
        scale_confidences = sorted(best_per_scale.values(), reverse=True)
        max_confidence = scale_confidences[0]
        if len(scale_confidences) >= 2:
            support = sum(scale_confidences[1:]) / len(scale_confidences[1:])
            confidence = min(1.0, max_confidence + consensus_boost * support)
        else:
            confidence = max_confidence * 0.90 if max_confidence < 0.11 else max_confidence
        fused.append(Detection(anchor.class_id, confidence, fused_xyxy, anchor.track_id))

    return fused


def predict_multiscale(
    detector: Any,
    image: Any,
    *,
    primary_size: int = 640,
    multiscale_sizes: Sequence[int] = (),
    candidate_confidence: float = 0.04,
    nms_iou: float = 0.65,
    agnostic_nms: bool = True,
    device: str | int = "cpu",
) -> list[Detection]:
    """Run YOLO inference at one or more scales and fuse the candidates."""
    scales: list[int] = []
    for size in (multiscale_sizes if multiscale_sizes else (primary_size,)):
        if size > 0 and size not in scales:
            scales.append(int(size))
    scales = scales or [primary_size]
    height, width = image.shape[:2]
    all_detections: list[tuple[Detection, int]] = []

    for size in scales:
        result = detector.predict(
            source=image,
            imgsz=size,
            conf=candidate_confidence,
            iou=nms_iou,
            agnostic_nms=agnostic_nms,
            device=device,
            save=False,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or not len(boxes):
            continue
        raw = [
            Detection(
                int(box.cls.item()),
                float(box.conf.item()),
                tuple(int(round(value)) for value in box.xyxy[0].tolist()),
            )
            for box in boxes
        ]
        all_detections.extend((detection, size) for detection in raw)

    return fuse_multiscale_detections(all_detections, num_scales=len(scales))


def generate_tiles(
    width: int,
    height: int,
    tile_size: int = 640,
    overlap: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate overlapping tiles covering an image."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    tile_width = min(width, max(64, tile_size))
    tile_height = min(height, max(64, tile_size))
    step_x = max(1, int(round(tile_width * (1.0 - overlap))))
    step_y = max(1, int(round(tile_height * (1.0 - overlap))))
    xs = list(range(0, max(1, width - tile_width + 1), step_x))
    ys = list(range(0, max(1, height - tile_height + 1), step_y))
    if not xs or xs[-1] + tile_width < width:
        xs.append(max(0, width - tile_width))
    if not ys or ys[-1] + tile_height < height:
        ys.append(max(0, height - tile_height))
    return [
        (x, y, min(width, x + tile_width), min(height, y + tile_height))
        for y in ys
        for x in xs
    ]
