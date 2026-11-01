"""Merge reviewed batches into a clean, auditable severity dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import PROJECT_ROOT
from src.severity.review import SEVERITY_LABELS, resolve_review_label


DEFAULT_INPUT = (
    PROJECT_ROOT / "outputs" / "pothole_review_batch3" / "pothole_labels.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Review manifest; repeat to merge batches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "severity_classifier",
    )
    parser.add_argument(
        "--require-two-reviewers",
        action="store_true",
        help="Exclude rows without consensus or an adjudicated label",
    )
    return parser


def prepare_dataset(
    inputs: list[Path],
    output_dir: Path,
    *,
    require_two_reviewers: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for input_path in inputs:
        path = input_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Review manifest not found: {path}")
        frame = pd.read_csv(path, dtype=str).fillna("")
        required = {"detection_id", "image_name", "crop_path"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        try:
            source_manifest = str(path.relative_to(project_root))
        except ValueError:
            source_manifest = str(path)
        frame["source_manifest"] = source_manifest
        if "batch_id" not in frame.columns:
            frame["batch_id"] = path.parent.name
        else:
            frame["batch_id"] = frame["batch_id"].replace("", path.parent.name)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    resolutions = combined.apply(
        lambda row: resolve_review_label(
            row, require_two_reviewers=require_two_reviewers
        ),
        axis=1,
    )
    combined["resolved_label"] = [item[0] for item in resolutions]
    combined["label_source"] = [item[1] for item in resolutions]
    combined["original_detection_id"] = combined["detection_id"].astype(str)
    duplicate_original_ids = combined["detection_id"].duplicated(keep=False)
    combined.loc[duplicate_original_ids, "detection_id"] = (
        combined.loc[duplicate_original_ids, "batch_id"].astype(str)
        + ":"
        + combined.loc[duplicate_original_ids, "original_detection_id"].astype(str)
    )

    remaining_duplicates = combined["detection_id"].duplicated(keep=False)
    if remaining_duplicates.any():
        examples = combined.loc[remaining_duplicates, "detection_id"].head(3).tolist()
        raise ValueError(f"Duplicate review identifiers after merge: {examples}")

    valid = set(SEVERITY_LABELS)
    clean = combined[combined["resolved_label"].isin(valid)].copy()
    clean["manual_label"] = clean["resolved_label"]
    score_mapping = {"low": 1, "medium": 2, "high": 3}
    clean["severity_score"] = clean["manual_label"].map(score_mapping).astype(int)
    excluded = combined[~combined.index.isin(clean.index)].copy()

    missing_crops = [
        value
        for value in clean["crop_path"]
        if not (project_root / str(value)).is_file()
    ]
    if missing_crops:
        raise FileNotFoundError(
            f"{len(missing_crops)} reviewed crops are missing; first: {missing_crops[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / "severity_dataset.csv", index=False)
    excluded.to_csv(output_dir / "excluded_detections.csv", index=False)
    return clean, excluded


def main() -> None:
    args = build_parser().parse_args()
    inputs = args.inputs or [DEFAULT_INPUT]
    clean, excluded = prepare_dataset(
        inputs,
        args.output.resolve(),
        require_two_reviewers=args.require_two_reviewers,
    )
    print(f"Review rows: {len(clean) + len(excluded)}")
    print(f"Usable labelled examples: {len(clean)}")
    print(f"Excluded or unresolved: {len(excluded)}")
    print(f"Unique source images: {clean['image_name'].nunique()}")
    print("Class distribution:")
    print(clean["manual_label"].value_counts().reindex(SEVERITY_LABELS, fill_value=0))
    print("Label provenance:")
    print(clean["label_source"].value_counts())


if __name__ == "__main__":
    main()
