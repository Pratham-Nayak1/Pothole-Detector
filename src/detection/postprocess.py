"""Class-aware confidence filtering and class-agnostic duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


def _intersection_and_areas(
    first: Detection, second: Detection
) -> tuple[float, float, float]:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = float(intersection_width * intersection_height)
    first_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    second_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    return intersection, first_area, second_area


def overlap_scores(first: Detection, second: Detection) -> tuple[float, float]:
    """Return IoU and intersection-over-smaller-box (IoS)."""
    intersection, first_area, second_area = _intersection_and_areas(first, second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller_area = min(first_area, second_area)
    ios = intersection / smaller_area if smaller_area > 0 else 0.0
    return iou, ios


def suppress_parasitic_boxes(detections: list[Detection]) -> list[Detection]:
    """Drop weak outer pothole boxes that contain a sharper inner detection."""
    kept: list[Detection] = []
    for index, outer in enumerate(detections):
        if outer.class_id != 3 or outer.confidence >= 0.42:
            kept.append(outer)
            continue
        ox1, oy1, ox2, oy2 = outer.xyxy
        outer_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
        parasitic = False
        for other_index, inner in enumerate(detections):
            if index == other_index or inner.class_id != 3:
                continue
            ix1, iy1, ix2, iy2 = inner.xyxy
            inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            _, ios = overlap_scores(outer, inner)
            if (
                outer_area > 2.0 * inner_area
                and ios >= 0.55
                and inner.confidence >= max(0.14, outer.confidence * 1.30)
            ):
                parasitic = True
                break
        if not parasitic:
            kept.append(outer)
    return kept


def _should_merge_same_class(
    first: Detection,
    second: Detection,
    class_id: int,
    merge_iou: float,
    merge_ios: float,
    img_width: int | None = None,
) -> bool:
    iou, ios = overlap_scores(first, second)
    if class_id == 3:
        # Potholes (D40): merge if there is any direct bounding box overlap
        if iou >= merge_iou or ios >= merge_ios:
            return True
        # Or if two sub-boxes are horizontally aligned along the same road plane (e.g. rims of a large pothole crater)
        ax1, ay1, ax2, ay2 = first.xyxy
        bx1, by1, bx2, by2 = second.xyxy
        y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
        min_h = min(ay2 - ay1, by2 - by1)
        if min_h > 0 and (y_overlap / min_h) >= 0.30:
            h_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
            max_gap = max(180, int(0.28 * img_width)) if img_width else 180
            if h_gap <= max_gap:
                return True
        return False
    else:
        # Cracks (0, 1, 2): merge on direct IoU overlap to preserve distinct crack lines
        return iou >= merge_iou


def merge_overlapping_same_class(
    detections: list[Detection],
    *,
    merge_iou: float = 0.15,
    merge_ios: float = 0.05,
    img_width: int | None = None,
) -> list[Detection]:
    """Union overlapping boxes of the same class into one encompassing bbox.

    Any two boxes of the same class that overlap or belong to the same physical
    defect are merged into a single box spanning their union. The merged box keeps
    the maximum confidence and track_id. Applied iteratively until convergence.
    """
    if not detections:
        return detections

    # Group by class
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    merged_all: list[Detection] = []
    for class_id, group in by_class.items():
        changed = True
        while changed:
            changed = False
            merged: list[Detection] = []
            used = [False] * len(group)
            for i, a in enumerate(group):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = a.xyxy
                best_conf = a.confidence
                best_tid = a.track_id
                for j, b in enumerate(group):
                    if i == j or used[j]:
                        continue
                    if _should_merge_same_class(
                        Detection(class_id, best_conf, (ax1, ay1, ax2, ay2)),
                        b,
                        class_id,
                        merge_iou,
                        merge_ios,
                        img_width,
                    ):
                        bx1, by1, bx2, by2 = b.xyxy
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        best_conf = max(best_conf, b.confidence)
                        if b.confidence > a.confidence:
                            best_tid = b.track_id
                        used[j] = True
                        changed = True
                merged.append(
                    Detection(
                        class_id=class_id,
                        confidence=best_conf,
                        xyxy=(ax1, ay1, ax2, ay2),
                        track_id=best_tid,
                    )
                )
            group = merged
        merged_all.extend(group)

    return merged_all


def filter_detections(
    detections: Iterable[Detection],
    *,
    default_confidence: float,
    class_confidence: Mapping[int, float],
    duplicate_iou: float,
    duplicate_ios: float,
    img_width: int | None = None,
    img_height: int | None = None,
) -> list[Detection]:
    """Apply per-class thresholds, merge overlapping same-class boxes, then
    remove same-object boxes across classes."""
    candidates = [
        detection
        for detection in detections
        if detection.confidence
        >= class_confidence.get(detection.class_id, default_confidence)
    ]
    candidates = suppress_parasitic_boxes(candidates)
    # Union fragmented boxes of the same class before cross-class dedup.
    candidates = merge_overlapping_same_class(
        candidates,
        merge_iou=0.15,
        merge_ios=0.05,
        img_width=img_width,
    )
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for accepted in kept:
            iou, ios = overlap_scores(candidate, accepted)
            if iou >= duplicate_iou or ios >= duplicate_ios:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept

