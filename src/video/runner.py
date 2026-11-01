"""Video inference with RoadTracker and track-level severity aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.common.config import PipelineSettings, resolve_device
from src.detection.postprocess import Detection, filter_detections
from src.pipeline.io import draw_detection_label, padded_crop
from src.pipeline.models import SeverityPredictor
from src.video.tracker import RoadTracker
from src.video.tracks import TrackSummary


def _ensure_browser_compatible_video(video_path: Path) -> None:
    """Transcode MP4 to H.264 (yuv420p + faststart) for seamless in-browser playback."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = video_path.with_name(f"{video_path.stem}_browser.mp4")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(temp_h264),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and temp_h264.exists() and temp_h264.stat().st_size > 0:
            video_path.unlink(missing_ok=True)
            temp_h264.replace(video_path)
    except Exception:
        # Fallback to original OpenCV file if ffmpeg is unavailable
        pass


POTHOLE_CLASS_ID = 3
TRACK_FIELDS = [
    "track_id",
    "class_id",
    "class_name",
    "first_frame",
    "last_frame",
    "first_time_seconds",
    "last_time_seconds",
    "observations",
    "max_detection_confidence",
    "severity",
    "severity_confidence",
    "severity_observations",
]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def run_video(
    input_path: Path,
    settings: PipelineSettings,
    *,
    output_dir: Path,
    severity_enabled: bool | None = None,
) -> dict:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")
    if not settings.detector.model_path.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {settings.detector.model_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*settings.video.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_video}")

    device = resolve_device(settings.detector.device)
    detector = YOLO(str(settings.detector.model_path))
    use_severity = (
        settings.severity.enabled if severity_enabled is None else severity_enabled
    )
    severity = (
        SeverityPredictor(
            settings.severity.model_path,
            settings.severity.classes,
            settings.severity.image_size,
            device,
        )
        if use_severity
        else None
    )
    road_tracker = RoadTracker(
        iou_threshold=0.15,
        max_center_distance=140.0,
        max_age_frames=30,
    )
    tracks: dict[int, TrackSummary] = {}
    untracked_detections = 0
    suppressed_or_filtered = 0
    processed_frames = 0
    started_at = datetime.now(timezone.utc)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = processed_frames
            processed_frames += 1

            result = detector.predict(
                source=frame,
                imgsz=settings.detector.image_size,
                conf=settings.detector.candidate_confidence,
                iou=settings.detector.nms_iou,
                agnostic_nms=settings.detector.agnostic_nms,
                device=device,
                save=False,
                verbose=False,
            )[0]
            annotated = frame.copy()
            boxes = result.boxes

            if boxes is not None and len(boxes):
                raw_detections: list[Detection] = [
                    Detection(
                        class_id=int(box.cls.item()),
                        confidence=float(box.conf.item()),
                        xyxy=tuple(
                            int(round(value)) for value in box.xyxy[0].tolist()
                        ),
                    )
                    for box in boxes
                ]
                detections = filter_detections(
                    raw_detections,
                    default_confidence=settings.detector.confidence,
                    class_confidence=settings.detector.class_confidence,
                    duplicate_iou=settings.detector.duplicate_iou,
                    duplicate_ios=settings.detector.duplicate_ios,
                    img_width=width,
                )
                suppressed_or_filtered += len(raw_detections) - len(detections)

                # Continuous spatial tracking ensures every detection is visible
                # from its very first appearance without initial-frame hiding
                tracked_detections = road_tracker.update(detections, frame_index)

                for detection in tracked_detections:
                    track_id = detection.track_id
                    if track_id is None:
                        continue
                    class_id = detection.class_id
                    confidence = detection.confidence
                    coordinates = detection.xyxy
                    summary = tracks.get(track_id)
                    if summary is None:
                        summary = TrackSummary(
                            track_id=track_id,
                            class_id=class_id,
                            class_name=_class_name(detector.names, class_id),
                            first_frame=frame_index,
                            last_frame=frame_index,
                        )
                        tracks[track_id] = summary
                    summary.update_detection(frame_index, confidence)

                    if (
                        class_id == POTHOLE_CLASS_ID
                        and severity is not None
                        and summary.should_sample_severity(
                            frame_index, settings.video.severity_interval_frames
                        )
                    ):
                        crop = padded_crop(frame, coordinates)
                        if crop.size:
                            _, _, probabilities = severity.predict_with_bbox(
                                crop,
                                coordinates,
                                (width, height),
                            )
                            summary.update_severity(frame_index, probabilities)

                    severity_label, severity_confidence = summary.severity()
                    draw_detection_label(
                        annotated,
                        coordinates,
                        class_id,
                        _class_name(detector.names, class_id),
                        confidence,
                        track_id=track_id,
                        severity=severity_label,
                        severity_confidence=(
                            severity_confidence if severity_label else None
                        ),
                    )

            writer.write(annotated)
            if processed_frames % 100 == 0:
                print(
                    f"Frames: {processed_frames}/{frame_count or '?'} | "
                    f"tracks: {len(tracks)}"
                )
    finally:
        capture.release()
        writer.release()

    # Transcode output video to web-standard H.264 so browsers can display it inline
    _ensure_browser_compatible_video(output_video)

    # Filter out transient flickering noise tracks (< 5 observations on real videos)
    # This prevents repetitive / noisy tracks from artificially inflating the defect count
    min_observations = max(2, min(5, int(0.04 * processed_frames)))
    confirmed_tracks = {
        tid: t for tid, t in tracks.items()
        if t.observations >= min_observations or processed_frames < 15
    }
    if not confirmed_tracks and tracks:
        # Fallback: preserve at least the primary track
        best_tid = max(tracks, key=lambda k: tracks[k].observations)
        confirmed_tracks = {best_tid: tracks[best_tid]}

    track_rows = [
        confirmed_tracks[track_id].as_dict(fps)
        for track_id in sorted(confirmed_tracks)
    ]
    tracks_csv = output_dir / "tracks.csv"
    with tracks_csv.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=TRACK_FIELDS)
        writer_csv.writeheader()
        writer_csv.writerows(track_rows)

    report = {
        "schema_version": 2,
        "run": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "input_video": str(input_path),
            "output_video": str(output_video),
            "fps": fps,
            "frames": processed_frames,
            "duration_seconds": processed_frames / fps if fps else None,
            "tracks": len(track_rows),
            "unique_pothole_tracks": sum(
                row["class_id"] == POTHOLE_CLASS_ID for row in track_rows
            ),
            "untracked_detections": untracked_detections,
            "suppressed_or_filtered_observations": suppressed_or_filtered,
        },
        "configuration": {
            "detector_model": str(settings.detector.model_path),
            "severity_model": str(settings.severity.model_path) if use_severity else None,
            "severity_enabled": use_severity,
            "device": str(device),
            "tracker": settings.video.tracker,
            "severity_interval_frames": settings.video.severity_interval_frames,
            "confidence": settings.detector.confidence,
            "candidate_confidence": settings.detector.candidate_confidence,
            "class_confidence": settings.detector.class_confidence,
            "nms_iou": settings.detector.nms_iou,
            "agnostic_nms": settings.detector.agnostic_nms,
            "duplicate_iou": settings.detector.duplicate_iou,
            "duplicate_ios": settings.detector.duplicate_ios,
        },
        "tracks": track_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
