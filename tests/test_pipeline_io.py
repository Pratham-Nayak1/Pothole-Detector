import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.io import (
    discover_images,
    output_image_path,
    padded_crop,
    write_reports,
)


def test_discover_images_supports_file_and_recursive_directory(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    first.write_bytes(b"image")
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.png"
    second.write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert discover_images(first) == [first]
    assert discover_images(tmp_path) == [first]
    assert discover_images(tmp_path, recursive=True) == [first, second]


def test_discover_images_rejects_unsupported_file(tmp_path: Path) -> None:
    file = tmp_path / "video.mp4"
    file.write_bytes(b"video")

    with pytest.raises(ValueError, match="Unsupported"):
        discover_images(file)


def test_padded_crop_clamps_to_image_boundaries() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    crop = padded_crop(image, (0, 0, 20, 20), padding=0.5)

    assert crop.shape == (30, 30, 3)


def test_output_image_path_preserves_nested_structure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    image = input_dir / "route-a" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    output = tmp_path / "output"

    destination = output_image_path(image, input_dir, output)

    assert destination == output / "images" / "route-a" / "frame.jpg"
    assert destination.parent.is_dir()


def test_write_reports_outputs_csv_and_json(tmp_path: Path) -> None:
    row = {
        "image": "frame.jpg",
        "detection_id": "I00001-D001",
        "class_id": 3,
        "class_name": "Pothole (D40)",
        "detection_confidence": 0.9,
        "x1": 1,
        "y1": 2,
        "x2": 30,
        "y2": 40,
        "severity": "high",
        "severity_confidence": 0.8,
        "severity_probabilities": {"low": 0.1, "medium": 0.1, "high": 0.8},
    }
    report = {"detections": [row]}

    csv_path, json_path = write_reports(
        tmp_path, [row], report, save_csv=True, save_json=True
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with csv_path.open(encoding="utf-8", newline="") as file:
        written = next(csv.DictReader(file))
    assert json.loads(written["severity_probabilities"])["high"] == 0.8
