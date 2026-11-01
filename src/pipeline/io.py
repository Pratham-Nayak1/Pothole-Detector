"""Image discovery, cropping, annotation, and report output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_FIELDS = [
    "image",
    "detection_id",
    "class_id",
    "class_name",
    "detection_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "severity",
    "severity_confidence",
    "severity_probabilities",
]


def discover_images(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return images


def padded_crop(image, box: tuple[int, int, int, int], padding: float = 0.20):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(10, int((x2 - x1) * padding))
    pad_y = max(10, int((y2 - y1) * padding))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    return image[top:bottom, left:right]


def output_image_path(image: Path, input_path: Path, output_dir: Path) -> Path:
    if input_path.is_file():
        relative = Path(image.name)
    else:
        relative = image.relative_to(input_path)
    destination = output_dir / "images" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def draw_severity_label(
    image,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    x1, y1, x2, y2 = box
    text = f"severity: {label} {confidence:.2f}"
    image_height, image_width = image.shape[:2]
    font_scale = max(0.55, min(image_height, image_width) / 1100.0)
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = max(0, min(x1, image_width - text_width - 8))
    text_y = y1 - 10
    if text_y - text_height < 0:
        text_y = min(image_height - baseline - 4, y2 + text_height + 10)

    cv2.rectangle(
        image,
        (text_x, text_y - text_height - 6),
        (min(image_width - 1, text_x + text_width + 8), text_y + baseline + 4),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        text,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_detection_label(
    image,
    box: tuple[int, int, int, int],
    class_id: int,
    class_name: str,
    confidence: float,
    *,
    track_id: int | None = None,
    severity: str = "",
    severity_confidence: float | None = None,
) -> None:
    """Draw one consolidated label for a post-processed detection."""
    colors = {
        0: (255, 170, 0),
        1: (255, 80, 80),
        2: (0, 200, 255),
        3: (0, 70, 255),
    }
    color = colors.get(class_id, (120, 255, 120))
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    prefix = f"#{track_id} " if track_id is not None else ""
    text = f"{prefix}{class_name} {confidence:.2f}"
    if severity and severity_confidence is not None:
        text += f" | severity {severity} {severity_confidence:.2f}"

    image_height, image_width = image.shape[:2]
    font_scale = max(0.45, min(image_height, image_width) / 1400.0)
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = max(0, min(x1, image_width - text_width - 8))
    text_y = y1 - 8
    if text_y - text_height < 0:
        text_y = min(image_height - baseline - 4, y1 + text_height + 10)
    cv2.rectangle(
        image,
        (text_x, text_y - text_height - 5),
        (min(image_width - 1, text_x + text_width + 8), text_y + baseline + 3),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def write_reports(
    output_dir: Path,
    rows: Iterable[dict[str, Any]],
    report: dict[str, Any],
    *,
    save_csv: bool,
    save_json: bool,
) -> tuple[Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    csv_path = output_dir / "detections.csv" if save_csv else None
    json_path = output_dir / "report.json" if save_json else None

    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in row_list:
                csv_row = dict(row)
                csv_row["severity_probabilities"] = json.dumps(
                    row.get("severity_probabilities", {}), sort_keys=True
                )
                writer.writerow(csv_row)

    if json_path:
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return csv_path, json_path
