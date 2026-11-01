"""Typed project configuration loaded from ``configs/pipeline.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class DetectorSettings:
    model_path: Path
    data_yaml: Path
    confidence: float
    candidate_confidence: float
    class_confidence: dict[int, float]
    nms_iou: float
    agnostic_nms: bool
    duplicate_iou: float
    duplicate_ios: float
    image_size: int
    device: str

    def confidence_for(self, class_id: int) -> float:
        """Return the final reporting threshold for a detector class."""
        return self.class_confidence.get(class_id, self.confidence)


@dataclass(frozen=True)
class SeveritySettings:
    enabled: bool
    model_path: Path
    image_size: int
    classes: tuple[str, ...]
    confidence_threshold: float = 0.55


@dataclass(frozen=True)
class OutputSettings:
    directory: Path
    save_images: bool
    save_csv: bool
    save_json: bool


@dataclass(frozen=True)
class VideoSettings:
    tracker: str
    severity_interval_frames: int
    codec: str


@dataclass(frozen=True)
class PipelineSettings:
    detector: DetectorSettings
    severity: SeveritySettings
    output: OutputSettings
    video: VideoSettings
    config_path: Path
    project_root: Path


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required setting: {section}.{key}")
    return mapping[key]


def load_settings(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> PipelineSettings:
    """Load and validate the shared pipeline configuration."""
    root = Path(project_root).resolve()
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    detector = raw.get("detector", {})
    severity = raw.get("severity", {})
    output = raw.get("output", {})
    video = raw.get("video", {})

    confidence = float(_require(detector, "confidence", "detector"))
    candidate_confidence = float(detector.get("candidate_confidence", confidence))
    nms_iou = float(_require(detector, "nms_iou", "detector"))
    duplicate_iou = float(detector.get("duplicate_iou", nms_iou))
    duplicate_ios = float(detector.get("duplicate_ios", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("detector.confidence must be between 0 and 1")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("detector.nms_iou must be between 0 and 1")
    for name, value in {
        "candidate_confidence": candidate_confidence,
        "duplicate_iou": duplicate_iou,
        "duplicate_ios": duplicate_ios,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"detector.{name} must be between 0 and 1")

    class_confidence = {
        int(class_id): float(value)
        for class_id, value in detector.get("class_confidence", {}).items()
    }
    if any(not 0.0 <= value <= 1.0 for value in class_confidence.values()):
        raise ValueError("detector.class_confidence values must be between 0 and 1")
    final_thresholds = [confidence, *class_confidence.values()]
    if candidate_confidence > min(final_thresholds):
        raise ValueError(
            "detector.candidate_confidence must not exceed a final confidence threshold"
        )

    classes = tuple(str(item) for item in severity.get("classes", []))
    if not classes:
        raise ValueError("severity.classes must contain at least one class")

    return PipelineSettings(
        detector=DetectorSettings(
            model_path=_resolve_path(
                str(_require(detector, "model_path", "detector")), root
            ),
            data_yaml=_resolve_path(
                str(_require(detector, "data_yaml", "detector")), root
            ),
            confidence=confidence,
            candidate_confidence=candidate_confidence,
            class_confidence=class_confidence,
            nms_iou=nms_iou,
            agnostic_nms=bool(detector.get("agnostic_nms", False)),
            duplicate_iou=duplicate_iou,
            duplicate_ios=duplicate_ios,
            image_size=int(_require(detector, "image_size", "detector")),
            device=str(detector.get("device", "auto")),
        ),
        severity=SeveritySettings(
            enabled=bool(severity.get("enabled", True)),
            model_path=_resolve_path(
                str(_require(severity, "model_path", "severity")), root
            ),
            image_size=int(severity.get("image_size", 224)),
            classes=classes,
            confidence_threshold=float(severity.get("confidence_threshold", 0.55)),
        ),
        output=OutputSettings(
            directory=_resolve_path(str(output.get("directory", "outputs/inference")), root),
            save_images=bool(output.get("save_images", True)),
            save_csv=bool(output.get("save_csv", True)),
            save_json=bool(output.get("save_json", True)),
        ),
        video=VideoSettings(
            tracker=str(video.get("tracker", "bytetrack.yaml")),
            severity_interval_frames=max(
                1, int(video.get("severity_interval_frames", 10))
            ),
            codec=str(video.get("codec", "mp4v")),
        ),
        config_path=path,
        project_root=root,
    )


def resolve_device(requested: str) -> str | int:
    """Translate ``auto`` into the best available Ultralytics device value."""
    if requested.lower() != "auto":
        return int(requested) if requested.isdigit() else requested

    import torch

    return 0 if torch.cuda.is_available() else "cpu"
