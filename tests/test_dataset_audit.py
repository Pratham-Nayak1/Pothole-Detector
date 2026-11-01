from pathlib import Path

from src.evaluation.dataset_audit import audit_dataset, parse_label_file


def create_split(root: Path, split: str, label: str = "3 0.5 0.5 0.2 0.2") -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / f"{split}_sample.jpg").write_bytes(b"test-image")
    (label_dir / f"{split}_sample.txt").write_text(label, encoding="utf-8")


def test_dataset_audit_accepts_valid_yolo_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    create_split(data_root, "train", "0 0.5 0.5 0.2 0.2")
    create_split(data_root, "val")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"""
path: {data_root.as_posix()}
train: images/train
val: images/val
nc: 4
names:
  0: D00
  1: D10
  2: D20
  3: D40
""".strip(),
        encoding="utf-8",
    )

    report = audit_dataset(data_yaml)

    assert report["valid"] is True
    assert report["splits"]["train"]["boxes"] == 1
    assert report["splits"]["val"]["class_counts"]["3"] == 1


def test_dataset_audit_reports_missing_labels(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_dir = data_root / "images" / "val"
    (data_root / "labels" / "val").mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (image_dir / "missing.jpg").write_bytes(b"image")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {data_root.as_posix()}\nval: images/val\nnc: 1\nnames: {{0: defect}}\n",
        encoding="utf-8",
    )

    report = audit_dataset(data_yaml)

    assert report["valid"] is False
    assert report["splits"]["val"]["missing_labels"] == ["missing"]


def test_parse_label_file_rejects_invalid_coordinates(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")

    counts, errors = parse_label_file(label, class_count=1)

    assert not counts
    assert "coordinates outside" in errors[0]


def test_parse_label_file_rejects_duplicate_rows(tmp_path: Path) -> None:
    label = tmp_path / "duplicate.txt"
    row = "0 0.5 0.5 0.2 0.2"
    label.write_text(f"{row}\n{row}\n", encoding="utf-8")

    counts, errors = parse_label_file(label, class_count=1)

    assert counts[0] == 1
    assert "duplicate annotation" in errors[0]
