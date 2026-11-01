"""Command-line entry point for the end-to-end image pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.common.config import PROJECT_ROOT, load_settings
from src.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image or directory")
    parser.add_argument("--output", type=Path, help="Output directory override")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
    )
    parser.add_argument("--device", help="Device override: auto, cpu, or GPU index")
    parser.add_argument(
        "--confidence", type=float, help="Global confidence override for every class"
    )
    parser.add_argument("--iou", type=float, help="NMS IoU threshold override")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-severity", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    detector = settings.detector

    if args.device is not None:
        detector = replace(detector, device=args.device)
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError("--confidence must be between 0 and 1")
        detector = replace(
            detector,
            confidence=args.confidence,
            candidate_confidence=args.confidence,
            class_confidence={},
        )
    if args.iou is not None:
        if not 0.0 <= args.iou <= 1.0:
            raise ValueError("--iou must be between 0 and 1")
        detector = replace(detector, nms_iou=args.iou)

    settings = replace(settings, detector=detector)
    report = run_pipeline(
        args.input,
        settings,
        output_dir=args.output,
        recursive=args.recursive,
        severity_enabled=False if args.no_severity else None,
    )
    print(
        f"Completed: {report['run']['images_processed']} images, "
        f"{report['run']['detections']} detections, "
        f"{report['run']['potholes']} potholes"
    )


if __name__ == "__main__":
    main()
