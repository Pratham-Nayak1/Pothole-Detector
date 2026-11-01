"""CLI for video detection, ByteTrack association, and severity aggregation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.common.config import PROJECT_ROOT, load_settings
from src.video.runner import run_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
    )
    parser.add_argument("--device", help="Device override: auto, cpu, or GPU index")
    parser.add_argument("--no-severity", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    if args.device is not None:
        settings = replace(
            settings,
            detector=replace(settings.detector, device=args.device),
        )
    report = run_video(
        args.input,
        settings,
        output_dir=args.output,
        severity_enabled=False if args.no_severity else None,
    )
    print(
        f"Completed: {report['run']['frames']} frames, "
        f"{report['run']['tracks']} tracks, "
        f"{report['run']['unique_pothole_tracks']} unique pothole tracks"
    )


if __name__ == "__main__":
    main()
