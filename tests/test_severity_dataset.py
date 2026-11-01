from pathlib import Path

import pandas as pd

from src.severity.dataset import create_grouped_splits, save_splits


def make_dataframe() -> pd.DataFrame:
    rows = []
    detection = 0
    for image_index in range(30):
        for label in ("low", "medium", "high"):
            detection += 1
            rows.append(
                {
                    "detection_id": f"P{detection:05d}",
                    "image_name": f"image_{image_index:03d}.jpg",
                    "crop_path": f"crops/P{detection:05d}.jpg",
                    "manual_label": label,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_splits_are_deterministic_and_isolated() -> None:
    dataframe = make_dataframe()

    first = create_grouped_splits(dataframe, seed=42)
    second = create_grouped_splits(dataframe, seed=42)

    assert [list(frame.index) for frame in first] == [
        list(frame.index) for frame in second
    ]
    group_sets = [set(frame["image_name"]) for frame in first]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert sum(len(frame) for frame in first) == len(dataframe)


def test_save_splits_records_counts(tmp_path: Path) -> None:
    train, validation, test = create_grouped_splits(make_dataframe(), seed=42)

    summary = save_splits(
        tmp_path, train, validation, test, seed=42
    )

    assert summary["splits"]["train"]["examples"] == len(train)
    assert (tmp_path / "train.csv").is_file()
    assert (tmp_path / "validation.csv").is_file()
    assert (tmp_path / "test.csv").is_file()
    assert (tmp_path / "split_summary.json").is_file()
