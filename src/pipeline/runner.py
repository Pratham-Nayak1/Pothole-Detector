"""Execution engine for end-to-end image inference."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.common.config import PipelineSettings, resolve_device
from src.detection.postprocess import Detection, filter_detections
from src.pipeline.io import (
    discover_images,
    draw_detection_label,
    output_image_path,
    padded_crop,
    write_reports,
)
from src.pipeline.models import SeverityPredictor


POTHOLE_CLASS_ID = 3


def _model_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def run_pipeline(
    input_path: Path,
    settings: PipelineSettings,
    *,
    output_dir: Path | None = None,
    recursive: bool = False,
    severity_enabled: bool | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    destination = (output_dir or settings.output.directory).resolve()
    use_severity = (
        settings.severity.enabled if severity_enabled is None else severity_enabled
    )

    if not settings.detector.model_path.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {settings.detector.model_path}"
        )

    images = discover_images(input_path, recursive=recursive)
    device = resolve_device(settings.detector.device)
    detector = YOLO(str(settings.detector.model_path))
    severity = None
    if use_severity:
        severity = SeverityPredictor(
            settings.severity.model_path,
            settings.severity.classes,
            settings.severity.image_size,
            device,
        )

    rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)

    for image_index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            image_summaries.append(
                {"image": str(image_path), "status": "unreadable", "detections": 0}
            )
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
        annotated = image.copy()
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
        detection_count = len(detections)

        for box_index, detection in enumerate(detections, start=1):
            class_id = detection.class_id
            confidence = detection.confidence
            coordinates = detection.xyxy
            x1, y1, x2, y2 = coordinates
            detection_id = f"I{image_index:05d}-D{box_index:03d}"
            severity_label = ""
            severity_confidence: float | str = ""
            severity_probabilities: dict[str, float] = {}

            if class_id == POTHOLE_CLASS_ID and severity is not None:
                crop = padded_crop(image, coordinates)
                if crop.size:
                    (
                        severity_label,
                        severity_confidence,
                        severity_probabilities,
                    ) = severity.predict(crop)
            draw_detection_label(
                annotated,
                coordinates,
                class_id,
                _model_name(detector.names, class_id),
                confidence,
                severity=severity_label,
                severity_confidence=(
                    float(severity_confidence) if severity_confidence != "" else None
                ),
            )

            row = {
                "image": str(image_path),
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": _model_name(detector.names, class_id),
                "detection_confidence": round(confidence, 6),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "severity": severity_label,
                "severity_confidence": (
                    round(float(severity_confidence), 6)
                    if severity_confidence != ""
                    else ""
                ),
                "severity_probabilities": severity_probabilities,
            }
            rows.append(row)

        if settings.output.save_images:
            image_output = output_image_path(image_path, input_path, destination)
            if not cv2.imwrite(str(image_output), annotated):
                raise OSError(f"Could not save annotated image: {image_output}")
        else:
            image_output = None

        image_summaries.append(
            {
                "image": str(image_path),
                "status": "processed",
                "detections": detection_count,
                "raw_candidates": len(raw_detections),
                "suppressed_or_filtered": len(raw_detections) - detection_count,
                "annotated_image": str(image_output) if image_output else None,
            }
        )
        print(f"[{image_index}/{len(images)}] {image_path.name}: {detection_count}")

    completed_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 2,
        "run": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "input": str(input_path),
            "output": str(destination),
            "images_discovered": len(images),
            "images_processed": sum(
                item["status"] == "processed" for item in image_summaries
            ),
            "detections": len(rows),
            "potholes": sum(row["class_id"] == POTHOLE_CLASS_ID for row in rows),
        },
        "configuration": {
            "detector_model": str(settings.detector.model_path),
            "severity_model": str(settings.severity.model_path) if use_severity else None,
            "severity_enabled": use_severity,
            "device": str(device),
            "image_size": settings.detector.image_size,
            "confidence": settings.detector.confidence,
            "candidate_confidence": settings.detector.candidate_confidence,
            "class_confidence": settings.detector.class_confidence,
            "nms_iou": settings.detector.nms_iou,
            "agnostic_nms": settings.detector.agnostic_nms,
            "duplicate_iou": settings.detector.duplicate_iou,
            "duplicate_ios": settings.detector.duplicate_ios,
        },
        "images": image_summaries,
        "detections": rows,
    }

    write_reports(
        destination,
        rows,
        report,
        save_csv=settings.output.save_csv,
        save_json=settings.output.save_json,
    )
    return report
