"""Prepare a resumable source-image batch for manual pothole review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.common.config import PROJECT_ROOT, load_settings, resolve_device
from src.detection.postprocess import Detection, filter_detections
from src.pipeline.io import draw_detection_label, padded_crop
from src.severity.review import (
    REVIEW_FIELDS,
    replace_with_retry,
    save_review_manifest,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POTHOLE_CLASS_ID = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "data" / "images" / "val"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "severity_review_250",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml"
    )
    parser.add_argument("--target-images", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="Device override: auto, cpu, or GPU index")
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"checked_images": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def _model_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.target_images < 1:
        raise ValueError("--target-images must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {source}")

    settings = load_settings(args.config)
    if not settings.detector.model_path.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {settings.detector.model_path}"
        )

    manifest_path = output / "pothole_labels.csv"
    state_path = output / "batch_state.json"
    crops_dir = output / "crops"
    annotated_dir = output / "annotated"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Review batch already exists: {manifest_path}. Use --resume to continue."
        )

    output.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(manifest_path)
    state = _load_state(state_path)
    checked_images = set(str(name) for name in state.get("checked_images", []))
    selected_images = {str(row["image_path"]) for row in rows}

    image_paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(args.seed).shuffle(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {source}")

    device = resolve_device(args.device or settings.detector.device)
    detector = YOLO(str(settings.detector.model_path))
    batch_id = output.name
    print(
        f"Preparing {args.target_images} source images for review; "
        f"already selected {len(selected_images)}."
    )

    for image_path in image_paths:
        relative_source = str(image_path.relative_to(PROJECT_ROOT))
        if relative_source in checked_images or relative_source in selected_images:
            continue
        if len(selected_images) >= args.target_images:
            break

        checked_images.add(relative_source)
        image = cv2.imread(str(image_path))
        if image is None:
            _save_state(state_path, {"checked_images": sorted(checked_images)})
            continue

        result = detector.predict(
            source=image,
            imgsz=settings.detector.image_size,
            conf=settings.detector.candidate_confidence,
            iou=settings.detector.nms_iou,
            agnostic_nms=settings.detector.agnostic_nms,
            device=device,
            save=False,
            verbose=False,
        )[0]
        boxes = result.boxes
        raw_detections = [
            Detection(
                class_id=int(box.cls.item()),
                confidence=float(box.conf.item()),
                xyxy=tuple(int(round(value)) for value in box.xyxy[0].tolist()),
            )
            for box in (boxes if boxes is not None else [])
        ]
        detections = filter_detections(
            raw_detections,
            default_confidence=settings.detector.confidence,
            class_confidence=settings.detector.class_confidence,
            duplicate_iou=settings.detector.duplicate_iou,
            duplicate_ios=settings.detector.duplicate_ios,
        )
        potholes = [item for item in detections if item.class_id == POTHOLE_CLASS_ID]
        if not potholes:
            _save_state(state_path, {"checked_images": sorted(checked_images)})
            continue

        annotated = image.copy()
        annotated_relative = Path("annotated") / image_path.relative_to(source)
        (output / annotated_relative).parent.mkdir(parents=True, exist_ok=True)
        for pothole in potholes:
            detection_number = len(rows) + 1
            detection_id = f"{batch_id}-P{detection_number:05d}"
            crop = padded_crop(image, pothole.xyxy)
            if not crop.size:
                continue
            crop_relative = Path("crops") / f"{detection_id}.jpg"
            crop_path = output / crop_relative
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not save review crop: {crop_path}")

            draw_detection_label(
                annotated,
                pothole.xyxy,
                pothole.class_id,
                _model_name(detector.names, pothole.class_id),
                pothole.confidence,
            )
            x1, y1, x2, y2 = pothole.xyxy
            row = {field: "" for field in REVIEW_FIELDS}
            row.update(
                {
                    "batch_id": batch_id,
                    "detection_id": detection_id,
                    "image_name": image_path.name,
                    "image_path": relative_source,
                    "crop_path": str((output / crop_relative).relative_to(PROJECT_ROOT)),
                    "annotated_path": str(
                        (output / annotated_relative).relative_to(PROJECT_ROOT)
                    ),
                    "confidence": round(pothole.confidence, 6),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            rows.append(row)

        if not cv2.imwrite(str(output / annotated_relative), annotated):
            raise OSError(f"Could not save annotated review image: {annotated_relative}")
        selected_images.add(relative_source)
        save_review_manifest(manifest_path, rows)
        _save_state(state_path, {"checked_images": sorted(checked_images)})
        print(
            f"Selected images: {len(selected_images)}/{args.target_images} | "
            f"checked: {len(checked_images)} | detections: {len(rows)}"
        )

    save_review_manifest(manifest_path, rows)
    state.update(
        {
            "batch_id": batch_id,
            "target_images": args.target_images,
            "selected_images": len(selected_images),
            "detections": len(rows),
            "seed": args.seed,
            "checked_images": sorted(checked_images),
            "complete": len(selected_images) >= args.target_images,
            "detector_model": str(settings.detector.model_path),
            "candidate_confidence": settings.detector.candidate_confidence,
            "pothole_confidence": settings.detector.confidence_for(POTHOLE_CLASS_ID),
        }
    )
    _save_state(state_path, state)
    print(f"Review manifest: {manifest_path}")
    print(
        "Start reviewer 1 with: streamlit run src/severity/review_app.py -- "
        f"--batch \"{output}\" --reviewer reviewer_1"
    )


if __name__ == "__main__":
    main()
