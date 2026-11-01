"""Audit YOLO image/label pairs and annotation validity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_definition(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    with data_yaml.open("r", encoding="utf-8") as file:
        definition = yaml.safe_load(file) or {}

    configured_root = Path(str(definition.get("path", data_yaml.parent)))
    if not configured_root.is_absolute():
        configured_root = (data_yaml.parent / configured_root).resolve()
    return configured_root, definition


def label_directory_for(image_directory: Path, dataset_root: Path) -> Path:
    try:
        relative = image_directory.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Image directory is outside dataset root: {image_directory}"
        ) from exc

    parts = list(relative.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer label directory from {image_directory}; "
            "expected an 'images' path component"
        ) from exc
    return dataset_root.joinpath(*parts)


def parse_label_file(path: Path, class_count: int) -> tuple[Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    seen_rows: set[str] = set()

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            errors.append(f"{path.name}:{line_number}: blank annotation row")
            continue
        if line in seen_rows:
            errors.append(f"{path.name}:{line_number}: duplicate annotation row")
            continue
        seen_rows.add(line)

        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"{path.name}:{line_number}: expected 5 fields, found {len(fields)}"
            )
            continue

        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_number}: non-numeric annotation")
            continue

        if not 0 <= class_id < class_count:
            errors.append(f"{path.name}:{line_number}: invalid class {class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            errors.append(f"{path.name}:{line_number}: coordinates outside [0, 1]")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path.name}:{line_number}: non-positive box size")
            continue

        counts[class_id] += 1

    return counts, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(
    image_directory: Path,
    label_directory: Path,
    class_count: int,
) -> tuple[dict[str, Any], list[Path]]:
    images = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(label_directory.glob("*.txt")) if label_directory.exists() else []

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    class_counts: Counter[int] = Counter()
    errors: list[str] = []

    for label in labels:
        counts, label_errors = parse_label_file(label, class_count)
        class_counts.update(counts)
        errors.extend(label_errors)

    result = {
        "image_directory": str(image_directory),
        "label_directory": str(label_directory),
        "images": len(images),
        "labels": len(labels),
        "boxes": sum(class_counts.values()),
        "class_counts": {
            str(index): class_counts[index] for index in range(class_count)
        },
        "missing_labels": sorted(image_stems - label_stems),
        "orphan_labels": sorted(label_stems - image_stems),
        "annotation_errors": errors,
    }
    return result, images


def audit_dataset(
    data_yaml: Path,
    *,
    check_content_hashes: bool = False,
) -> dict[str, Any]:
    dataset_root, definition = load_dataset_definition(data_yaml)
    names = definition.get("names", {})
    class_count = int(definition.get("nc", len(names)))
    split_results: dict[str, Any] = {}
    split_images: dict[str, list[Path]] = {}

    for split in ("train", "val", "test"):
        configured = definition.get(split)
        if not configured:
            continue
        image_directory = dataset_root / str(configured)
        if not image_directory.is_dir():
            split_results[split] = {
                "image_directory": str(image_directory),
                "error": "image directory does not exist",
            }
            continue
        label_directory = label_directory_for(image_directory, dataset_root)
        split_results[split], split_images[split] = audit_split(
            image_directory, label_directory, class_count
        )

    duplicates: list[dict[str, str]] = []
    if check_content_hashes:
        seen: dict[str, tuple[str, Path]] = {}
        for split, images in split_images.items():
            for image in images:
                content_hash = _sha256(image)
                if content_hash in seen and seen[content_hash][0] != split:
                    previous_split, previous_path = seen[content_hash]
                    duplicates.append(
                        {
                            "sha256": content_hash,
                            "first_split": previous_split,
                            "first_path": str(previous_path),
                            "second_split": split,
                            "second_path": str(image),
                        }
                    )
                else:
                    seen[content_hash] = (split, image)

    has_errors = any(
        result.get("error")
        or result.get("missing_labels")
        or result.get("orphan_labels")
        or result.get("annotation_errors")
        for result in split_results.values()
    )

    return {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "class_count": class_count,
        "class_names": names,
        "splits": split_results,
        "cross_split_exact_duplicates": duplicates,
        "content_hash_check_performed": check_content_hashes,
        "valid": not has_errors and not duplicates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="YOLO dataset YAML file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "dataset_audit.json",
        help="Destination JSON report",
    )
    parser.add_argument(
        "--check-content-hashes",
        action="store_true",
        help="Hash all images to detect exact duplicates across splits (slow)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(
        args.data.resolve(), check_content_hashes=args.check_content_hashes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset valid: {report['valid']}")
    for split, result in report["splits"].items():
        if "error" in result:
            print(f"{split}: {result['error']}")
            continue
        print(
            f"{split}: {result['images']} images, "
            f"{result['labels']} labels, {result['boxes']} boxes"
        )
    print(f"Report: {args.output}")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
