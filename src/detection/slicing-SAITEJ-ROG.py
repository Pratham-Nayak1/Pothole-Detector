"""Multi-scale pyramid inference, tiled slicing, and Confidence-Weighted Boxes Fusion (WBF)."""

from __future__ import annotations

from typing import Any, Sequence

from src.detection.postprocess import Detection, overlap_scores, suppress_parasitic_boxes


def fuse_multiscale_detections(
    scale_detections: Sequence[ tuple[Detection, int] ],
    *,
    fusion_iou: float = 0.40,
    consensus_boost: float = 0.22,
    num_scales: int = 1,
) -> list[Detection]:
    """Fuse detections across multiple inference scales using Confidence-Weighted Boxes Fusion (WBF).

    Detections confirmed across multiple scales receive a calibrated consensus confidence boost,
    while coordinates are computed as the squared-confidence-weighted average of matching boxes.
    """
    if not scale_detections:
        return []
    if num_scales <= 1:
        return [det for det, _ in scale_detections]

    ordered = sorted(scale_detections, key=lambda item: item[0].confidence, reverse=True)
    used = [False] * len(ordered)
    fused: list[Detection] = []

    for i, (anchor, anchor_scale) in enumerate(ordered):
        if used[i]:
            continue
        used[i] = True
        cluster: list[tuple[Detection, int]] = [(anchor, anchor_scale)]

        for j in range(i + 1, len(ordered)):
            if used[j]:
                continue
            cand, cand_scale = ordered[j]
            if cand.class_id != anchor.class_id:
                continue
            iou, ios = overlap_scores(anchor, cand)
            ax1, ay1, ax2, ay2 = anchor.xyxy
            bx1, by1, bx2, by2 = cand.xyxy
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            area_ratio = min(area_a, area_b) / max(area_a, area_b)

            if iou >= fusion_iou or (ios >= 0.72 and area_ratio >= 0.35):
                cluster.append((cand, cand_scale))
                used[j] = True

        # Weighted coordinate average (w = conf^2)
        weights = [max(1e-4, det.confidence ** 2) for det, _ in cluster]
        w_sum = sum(weights)
        fused_xyxy = tuple(
            int(round(sum(w * det.xyxy[k] for w, (det, _) in zip(weights, cluster)) / w_sum))
            for k in range(4)
        )

        # Best confidence per unique scale in this cluster
        best_per_scale: dict[int, float] = {}
        for det, sc in cluster:
            if sc not in best_per_scale or det.confidence > best_per_scale[sc]:
                best_per_scale[sc] = det.confidence

        scale_confs = sorted(best_per_scale.values(), reverse=True)
        max_conf = scale_confs[0]
        if len(scale_confs) >= 2:
            # Multi-scale agreement boost
            support = sum(scale_confs[1:]) / len(scale_confs[1:])
            final_conf = min(1.0, max_conf + consensus_boost * support)
        else:
            # Single-scale candidate: slightly dampen very weak borderline noise (< 0.11)
            final_conf = max_conf * 0.90 if max_conf < 0.11 else max_conf

        fused.append(
            Detection(
                class_id=anchor.class_id,
                confidence=final_conf,
                xyxy=fused_xyxy,
                track_id=anchor.track_id,
            )
        )

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
    """Run YOLO detector at one or more scales and fuse candidates with WBF."""
    scales: list[int] = []
    for sz in (multiscale_sizes if multiscale_sizes else (primary_size,)):
        if sz > 0 and sz not in scales:
            scales.append(int(sz))
    if not scales:
        scales = [primary_size]

    h, w = image.shape[:2]
    all_scale_detections: list[tuple[Detection, int]] = []

    for sz in scales:
        result = detector.predict(
            source=image,
            imgsz=sz,
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

        raw_scale: list[Detection] = [
            Detection(
                class_id=int(box.cls.item()),
                confidence=float(box.conf.item()),
                xyxy=tuple(int(round(value)) for value in box.xyxy[0].tolist()),
            )
            for box in boxes
        ]
        # Suppress giant low-confidence background hallucinations per scale before WBF
        cleaned_scale = suppress_parasitic_boxes(
            raw_scale,
            img_width=w,
            img_height=h,
            image=image,
        )
        for det in cleaned_scale:
            all_scale_detections.append((det, sz))

    return fuse_multiscale_detections(
        all_scale_detections,
        num_scales=len(scales),
    )


def generate_tiles(
    width: int,
    height: int,
    tile_size: int = 640,
    overlap: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate overlapping (x1, y1, x2, y2) tiles covering a high-resolution image."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    tile_w = min(width, max(64, tile_size))
    tile_h = min(height, max(64, tile_size))
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))

    xs = list(range(0, max(1, width - tile_w + 1), step_x))
    if not xs or xs[-1] + tile_w < width:
        xs.append(max(0, width - tile_w))

    ys = list(range(0, max(1, height - tile_h + 1), step_y))
    if not ys or ys[-1] + tile_h < height:
        ys.append(max(0, height - tile_h))

    return [(x, y, min(width, x + tile_w), min(height, y + tile_h)) for y in ys for x in xs]
