"""Reusable severity-dataset validation and grouped splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CLASSES = ("low", "medium", "high")
REQUIRED_COLUMNS = {"detection_id", "image_name", "crop_path", "manual_label"}


def load_clean_dataset(csv_path: Path, project_root: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Severity dataset CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Severity dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["manual_label"] = (
        dataframe["manual_label"].fillna("").str.lower().str.strip()
    )
    invalid_labels = sorted(set(dataframe["manual_label"]) - set(CLASSES))
    if invalid_labels:
        raise ValueError(f"Invalid severity labels: {invalid_labels}")
    if dataframe["detection_id"].duplicated().any():
        raise ValueError("Severity dataset contains duplicate detection_id values")

    missing_crops = [
        str(path)
        for path in dataframe["crop_path"]
        if not (project_root / str(path)).is_file()
    ]
    if missing_crops:
        raise FileNotFoundError(
            f"{len(missing_crops)} severity crops are missing; "
            f"first missing crop: {missing_crops[0]}"
        )
    return dataframe


def create_grouped_splits(
    dataframe: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    validation_size_within_train: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by source image so related crops cannot leak across partitions."""
    if dataframe["image_name"].nunique() < 3:
        raise ValueError("At least three unique source images are required")

    train_indices, test_indices = _best_group_split(
        dataframe, test_size=test_size, seed=seed
    )
    train_pool = dataframe.iloc[train_indices].copy()
    test = dataframe.iloc[test_indices].copy()

    train_sub_indices, validation_indices = _best_group_split(
        train_pool,
        test_size=validation_size_within_train,
        seed=seed + 1,
    )
    train = train_pool.iloc[train_sub_indices].copy()
    validation = train_pool.iloc[validation_indices].copy()
    validate_group_isolation(train, validation, test)
    return train, validation, test


def _best_group_split(
    dataframe: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    candidates: int = 250,
) -> tuple[list[int], list[int]]:
    """Choose a deterministic group split with the closest class proportions."""
    splitter = GroupShuffleSplit(
        n_splits=candidates, test_size=test_size, random_state=seed
    )
    overall = (
        dataframe["manual_label"].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    )
    class_counts = dataframe["manual_label"].value_counts().reindex(CLASSES, fill_value=0)
    best_score = float("inf")
    best: tuple[list[int], list[int]] | None = None
    for train_indices, held_indices in splitter.split(
        dataframe, groups=dataframe["image_name"]
    ):
        held = dataframe.iloc[held_indices]
        proportions = (
            held["manual_label"]
            .value_counts(normalize=True)
            .reindex(CLASSES, fill_value=0)
        )
        size_error = abs(len(held) / len(dataframe) - test_size)
        distribution_error = float((proportions - overall).abs().sum())
        missing_penalty = sum(
            5.0
            for label in CLASSES
            if class_counts[label] >= 3 and not (held["manual_label"] == label).any()
        )
        score = size_error * 2.0 + distribution_error + missing_penalty
        if score < best_score:
            best_score = score
            best = (train_indices.tolist(), held_indices.tolist())
    if best is None:
        raise ValueError("Could not construct a grouped severity split")
    return best


def validate_group_isolation(*dataframes: pd.DataFrame) -> None:
    group_sets = [set(frame["image_name"]) for frame in dataframes]
    for first_index, first in enumerate(group_sets):
        for second in group_sets[first_index + 1 :]:
            overlap = first & second
            if overlap:
                raise ValueError(
                    f"Source-image leakage across severity splits: {sorted(overlap)[:3]}"
                )


def save_splits(
    output_dir: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "validation": validation, "test": test}
    summary = {"seed": seed, "classes": list(CLASSES), "splits": {}}

    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "examples": len(frame),
            "source_images": int(frame["image_name"].nunique()),
            "class_counts": {
                label: int((frame["manual_label"] == label).sum())
                for label in CLASSES
            },
        }

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
