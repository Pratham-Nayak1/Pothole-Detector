"""Shared manifest schema and safe persistence for manual severity review."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SEVERITY_LABELS = ("low", "medium", "high")
REVIEW_LABELS = (*SEVERITY_LABELS, "uncertain", "duplicate", "not_pothole")
REVIEW_FIELDS = [
    "batch_id",
    "detection_id",
    "image_name",
    "image_path",
    "crop_path",
    "annotated_path",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "reviewer_1_label",
    "reviewer_1_notes",
    "reviewer_2_label",
    "reviewer_2_notes",
    "adjudicated_label",
    "adjudication_notes",
    "duplicate_of",
]


def replace_with_retry(temporary: Path, destination: Path) -> None:
    """Replace a file atomically, tolerating short Windows scanner locks."""
    for attempt in range(5):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def save_review_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically save a review manifest so interruption cannot erase progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
    replace_with_retry(temporary, path)


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_review_label(
    row: Mapping[str, Any], *, require_two_reviewers: bool = False
) -> tuple[str, str]:
    """Resolve a final label without hiding disagreement or exclusions."""
    legacy = normalize_label(row.get("manual_label", ""))
    if legacy:
        return legacy, "legacy_manual"

    adjudicated = normalize_label(row.get("adjudicated_label", ""))
    if adjudicated:
        return adjudicated, "adjudicated"

    first = normalize_label(row.get("reviewer_1_label", ""))
    second = normalize_label(row.get("reviewer_2_label", ""))
    if first and second:
        if first == second:
            return first, "reviewer_consensus"
        return "", "reviewer_disagreement"
    if require_two_reviewers:
        return "", "awaiting_second_review"
    if first:
        return first, "reviewer_1"
    if second:
        return second, "reviewer_2"
    return "", "unreviewed"
