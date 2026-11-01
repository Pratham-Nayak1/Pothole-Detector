from pathlib import Path

import pandas as pd

from src.severity.prepare_dataset import prepare_dataset
from src.severity.review import (
    REVIEW_FIELDS,
    resolve_review_label,
    save_review_manifest,
)


def test_resolve_review_label_preserves_disagreement_and_adjudication() -> None:
    disagreement = {
        "reviewer_1_label": "low",
        "reviewer_2_label": "high",
        "adjudicated_label": "",
    }
    assert resolve_review_label(disagreement) == ("", "reviewer_disagreement")

    disagreement["adjudicated_label"] = "medium"
    assert resolve_review_label(disagreement) == ("medium", "adjudicated")


def test_save_review_manifest_round_trips_schema(tmp_path: Path) -> None:
    path = tmp_path / "pothole_labels.csv"
    row = {field: "" for field in REVIEW_FIELDS}
    row.update({"batch_id": "batch", "detection_id": "P1", "image_name": "a.jpg"})

    save_review_manifest(path, [row])

    saved = pd.read_csv(path, dtype=str).fillna("")
    assert list(saved.columns) == REVIEW_FIELDS
    assert saved.iloc[0]["detection_id"] == "P1"
    assert not path.with_suffix(".tmp").exists()


def test_prepare_dataset_merges_legacy_and_review_batches(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy"
    review_dir = tmp_path / "review"
    legacy_dir.mkdir()
    review_dir.mkdir()
    crops = tmp_path / "crops"
    crops.mkdir()
    for name in ("legacy.jpg", "review.jpg", "excluded.jpg"):
        (crops / name).write_bytes(b"image")

    legacy = pd.DataFrame(
        [
            {
                "detection_id": "P1",
                "image_name": "legacy-source.jpg",
                "crop_path": "crops/legacy.jpg",
                "manual_label": "low",
            }
        ]
    )
    legacy_path = legacy_dir / "pothole_labels.csv"
    legacy.to_csv(legacy_path, index=False)

    review = pd.DataFrame(
        [
            {
                "batch_id": "batch2",
                "detection_id": "P1",
                "image_name": "review-source.jpg",
                "crop_path": "crops/review.jpg",
                "reviewer_1_label": "high",
                "reviewer_2_label": "high",
            },
            {
                "batch_id": "batch2",
                "detection_id": "P2",
                "image_name": "excluded-source.jpg",
                "crop_path": "crops/excluded.jpg",
                "reviewer_1_label": "uncertain",
                "reviewer_2_label": "uncertain",
            },
        ]
    )
    review_path = review_dir / "pothole_labels.csv"
    review.to_csv(review_path, index=False)

    clean, excluded = prepare_dataset(
        [legacy_path, review_path],
        tmp_path / "output",
        project_root=tmp_path,
    )

    assert clean["manual_label"].tolist() == ["low", "high"]
    assert clean["detection_id"].is_unique
    assert excluded["resolved_label"].tolist() == ["uncertain"]
