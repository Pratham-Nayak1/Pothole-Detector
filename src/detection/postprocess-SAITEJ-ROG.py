"""Class-aware confidence filtering, photometric validation, and duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


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


def is_road_paint_false_positive(
    image: Any,
    xyxy: tuple[int, int, int, int],
    confidence: float,
) -> bool:
    """Detect high-albedo white painted road markings misclassified as potholes."""
    if image is None or confidence >= 0.30:
        return False
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, min(w, xyxy[0])),
        max(0, min(h, xyxy[1])),
        max(0, min(w, xyxy[2])),
        max(0, min(h, xyxy[3])),
    )
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    p10 = float(np.percentile(gray, 10))
    mean_val = float(gray.mean())
    if p10 >= 150.0 and mean_val >= 175.0:
        return True
    if crop.ndim == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bright_white_frac = float(np.mean((gray > 195) & (hsv[:, :, 1] < 45)))
        if bright_white_frac >= 0.35 and p10 >= 135.0:
            return True
    return False


def _is_crater_bridge_continuous(
    image: Any,
    first: Detection,
    second: Detection,
) -> bool:
    """Verify whether the horizontal gap between two aligned pothole boxes is a continuous crater."""
    if image is None:
        return False
    h, w = image.shape[:2]
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    left_box, right_box = (
        (first.xyxy, second.xyxy) if ax1 <= bx1 else (second.xyxy, first.xyxy)
    )
    lx1, ly1, lx2, ly2 = left_box
    rx1, ry1, rx2, ry2 = right_box
    gx1 = max(0, min(w, lx2))
    gx2 = max(0, min(w, rx1))
    gy1 = max(0, min(h, max(ly1, ry1)))
    gy2 = max(0, min(h, min(ly2, ry2)))
    if gx2 - gx1 < 4 or gy2 - gy1 < 6:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bridge = gray[gy1:gy2, gx1:gx2]
    left_patch = gray[max(0, ly1) : min(h, ly2), max(0, lx1) : min(w, lx2)]
    right_patch = gray[max(0, ry1) : min(h, ry2), max(0, rx1) : min(w, rx2)]
    if bridge.size == 0 or left_patch.size == 0 or right_patch.size == 0:
        return False

    bridge_mean = float(bridge.mean())
    rim_mean = max(float(left_patch.mean()), float(right_patch.mean()))

    # Measure surrounding road surface above/below the candidate crater
    union_y1 = min(ly1, ry1)
    union_y2 = max(ly2, ry2)
    union_x1 = max(0, min(lx1, rx1))
    union_x2 = min(w, max(lx2, rx2))
    band_h = max(20, int(0.4 * (union_y2 - union_y1)))
    top_band = gray[max(0, union_y1 - band_h) : max(0, union_y1), union_x1:union_x2]
    bot_band = gray[min(h, union_y2) : min(h, union_y2 + band_h), union_x1:union_x2]
    context_pixels = [b.ravel() for b in (top_band, bot_band) if b.size > 0]
    if not context_pixels:
        return False
    road_mean = float(np.median(np.concatenate(context_pixels)))

    # A true crater floor between rims is significantly darker than intact surrounding road
    return bridge_mean <= road_mean - 18.0 and bridge_mean <= rim_mean + 8.0


def suppress_parasitic_boxes(
    detections: list[Detection],
    *,
    img_width: int | None = None,
    img_height: int | None = None,
    image: Any = None,
) -> list[Detection]:
    """Remove giant low-confidence background hallucinations and road-paint false positives."""
    if not detections:
        return detections

    if image is not None:
        img_height, img_width = image.shape[:2]

    frame_area = (
        float(img_width * img_height)
        if (img_width and img_height and img_width > 0 and img_height > 0)
        else None
    )

    plausible: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        if bw < 5 or bh < 5:
            continue
        if det.class_id == 3:
            if img_width and img_width > 0 and (bw / img_width) > 0.70 and det.confidence < 0.25:
                continue
            if frame_area and ((bw * bh) / frame_area) > 0.20 and det.confidence < 0.25:
                continue
            if image is not None and is_road_paint_false_positive(image, det.xyxy, det.confidence):
                continue
        plausible.append(det)

    # Suppress large, weaker outer pothole boxes that wrap sharper, higher-confidence inner pothole boxes
    kept: list[Detection] = []
    for i, outer in enumerate(plausible):
        if outer.class_id != 3 or outer.confidence >= 0.42:
            kept.append(outer)
            continue
        _, outer_area, _ = _intersection_and_areas(outer, outer)
        is_parasitic = False
        for j, inner in enumerate(plausible):
            if i == j or inner.class_id != 3:
                continue
            iou, ios = overlap_scores(outer, inner)
            _, inner_area, _ = _intersection_and_areas(inner, inner)
            if (
                inner_area > 0
                and outer_area > 2.0 * inner_area
                and ios >= 0.55
                and inner.confidence >= max(0.14, outer.confidence * 1.30)
            ):
                is_parasitic = True
                break
        if not is_parasitic:
            kept.append(outer)
    return kept


def _should_merge_same_class(
    first: Detection,
    second: Detection,
    class_id: int,
    merge_iou: float,
    merge_ios: float,
    img_width: int | None = None,
    image: Any = None,
) -> bool:
    iou, ios = overlap_scores(first, second)
    _, first_area, second_area = _intersection_and_areas(first, second)
    max_area = max(first_area, second_area)
    min_area = min(first_area, second_area)
    area_ratio = min_area / max_area if max_area > 0 else 0.0

    if class_id == 3:
        # Direct overlap between comparable pothole detections
        if iou >= merge_iou:
            return True
        # High containment where the larger box is confident or comparable in scale
        if ios >= merge_ios and (
            area_ratio >= 0.35
            or (first_area >= second_area and first.confidence >= second.confidence * 0.85)
            or (second_area >= first_area and second.confidence >= first.confidence * 0.85)
        ):
            return True
        # Horizontally aligned crater rims: only merge when verified by crater floor continuity
        ax1, ay1, ax2, ay2 = first.xyxy
        bx1, by1, bx2, by2 = second.xyxy
        y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
        ha, hb = max(0, ay2 - ay1), max(0, by2 - by1)
        min_h, max_h = min(ha, hb), max(ha, hb)
        wa, wb = max(0, ax2 - ax1), max(0, bx2 - bx1)
        if min_h > 0 and max_h > 0 and (y_overlap / min_h) >= 0.45 and (min_h / max_h) >= 0.35:
            h_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
            max_rim_gap = int(1.25 * max(wa, wb))
            union_w = max(ax2, bx2) - min(ax1, bx1)
            width_ok = (img_width is None) or (union_w <= int(0.88 * img_width))
            if h_gap <= max_rim_gap and width_ok and _is_crater_bridge_continuous(image, first, second):
                return True
        return False
    else:
        # Cracks (0, 1, 2): merge only on direct IoU overlap to preserve distinct crack lines
        return iou >= merge_iou


def _combine_same_class_pair(first: Detection, second: Detection) -> Detection:
    """Combine two overlapping same-class detections without allowing weak boxes to balloon tight ones."""
    iou, ios = overlap_scores(first, second)
    _, area_a, area_b = _intersection_and_areas(first, second)
    best_conf = max(first.confidence, second.confidence)
    best_tid = first.track_id if first.confidence >= second.confidence else second.track_id

    # If a much weaker sub-box is inside a stronger box, keep the stronger box's tight bounds
    if ios >= 0.65 and area_a > area_b * 1.6 and first.confidence >= second.confidence * 1.25:
        return Detection(first.class_id, best_conf, first.xyxy, best_tid)
    if ios >= 0.65 and area_b > area_a * 1.6 and second.confidence >= first.confidence * 1.25:
        return Detection(second.class_id, best_conf, second.xyxy, best_tid)

    # If two boxes strongly overlap on the same object, use confidence-weighted box refinement
    if iou >= 0.50:
        wa = first.confidence ** 2
        wb = second.confidence ** 2
        w_sum = wa + wb
        if w_sum > 0:
            xyxy = tuple(
                int(round((wa * first.xyxy[k] + wb * second.xyxy[k]) / w_sum))
                for k in range(4)
            )
            return Detection(first.class_id, best_conf, xyxy, best_tid)

    # True crater rim / fragment union
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    return Detection(
        class_id=first.class_id,
        confidence=best_conf,
        xyxy=(min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)),
        track_id=best_tid,
    )


def merge_overlapping_same_class(
    detections: list[Detection],
    *,
    merge_iou: float = 0.22,
    merge_ios: float = 0.60,
    img_width: int | None = None,
    image: Any = None,
) -> list[Detection]:
    """Consolidate overlapping or rim-bridged detections of the same class."""
    if not detections:
        return detections

    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    merged_all: list[Detection] = []
    for class_id, group in by_class.items():
        group = sorted(group, key=lambda item: item.confidence, reverse=True)
        changed = True
        while changed:
            changed = False
            merged: list[Detection] = []
            used = [False] * len(group)
            for i, a in enumerate(group):
                if used[i]:
                    continue
                current = a
                used[i] = True
                for j, b in enumerate(group):
                    if used[j]:
                        continue
                    if _should_merge_same_class(
                        current,
                        b,
                        class_id,
                        merge_iou,
                        merge_ios,
                        img_width,
                        image,
                    ):
                        current = _combine_same_class_pair(current, b)
                        used[j] = True
                        changed = True
                merged.append(current)
            group = sorted(merged, key=lambda item: item.confidence, reverse=True)
        merged_all.extend(group)

    return merged_all


def _are_perpendicular_crossing_cracks(first: Detection, second: Detection, iou: float) -> bool:
    """Return True if two crack boxes are perpendicular (e.g. longitudinal + transverse crossing)."""
    if first.class_id == 3 or second.class_id == 3 or iou >= 0.20:
        return False
    aw = max(1, first.xyxy[2] - first.xyxy[0])
    ah = max(1, first.xyxy[3] - first.xyxy[1])
    bw = max(1, second.xyxy[2] - second.xyxy[0])
    bh = max(1, second.xyxy[3] - second.xyxy[1])
    aspect_a = aw / ah
    aspect_b = bw / bh
    return (aspect_a >= 2.2 and aspect_b <= 0.45) or (aspect_b >= 2.2 and aspect_a <= 0.45)


def filter_detections(
    detections: Iterable[Detection],
    *,
    default_confidence: float,
    class_confidence: Mapping[int, float],
    duplicate_iou: float,
    duplicate_ios: float,
    img_width: int | None = None,
    img_height: int | None = None,
    image: Any = None,
) -> list[Detection]:
    """Apply per-class thresholds, suppress parasitic boxes, merge same-class fragments,
    and remove cross-class duplicates."""
    candidates = [
        detection
        for detection in detections
        if detection.confidence
        >= class_confidence.get(detection.class_id, default_confidence)
    ]
    candidates = suppress_parasitic_boxes(
        candidates,
        img_width=img_width,
        img_height=img_height,
        image=image,
    )
    candidates = merge_overlapping_same_class(
        candidates,
        merge_iou=0.22,
        merge_ios=0.60,
        img_width=img_width,
        image=image,
    )
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for accepted in kept:
            iou, ios = overlap_scores(candidate, accepted)
            if iou >= duplicate_iou:
                duplicate = True
                break
            if ios >= duplicate_ios and not _are_perpendicular_crossing_cracks(candidate, accepted, iou):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept
