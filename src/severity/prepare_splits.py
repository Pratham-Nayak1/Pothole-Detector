"""Validate the severity dataset and create fixed, group-isolated splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.severity.dataset import (
    create_grouped_splits,
    load_clean_dataset,
    save_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "severity_classifier" / "severity_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "severity_classifier" / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataframe = load_clean_dataset(args.dataset, PROJECT_ROOT)
    train, validation, test = create_grouped_splits(dataframe, seed=args.seed)
    summary = save_splits(
        args.output, train, validation, test, seed=args.seed
    )
    print(json.dumps(summary, indent=2))
    print(f"Fixed splits: {args.output}")


if __name__ == "__main__":
    main()
