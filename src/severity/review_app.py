"""Streamlit application for resumable, two-reviewer severity labelling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.common.config import PROJECT_ROOT
from src.severity.review import REVIEW_LABELS, save_review_manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batch",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "severity_review_250",
    )
    parser.add_argument(
        "--reviewer",
        choices=("reviewer_1", "reviewer_2", "adjudicator"),
        default="reviewer_1",
    )
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=100.0,
        help="Reviewer-2 percentage, sampled within reviewer-1 labels",
    )
    parser.add_argument("--seed", type=int, default=2026)
    arguments, _ = parser.parse_known_args()
    return arguments


def _read_manifest(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return pd.DataFrame(rows).fillna("")


def _save(dataframe: pd.DataFrame, path: Path) -> None:
    save_review_manifest(path, dataframe.to_dict(orient="records"))


args = _arguments()
batch = args.batch.resolve()
manifest_path = batch / "pothole_labels.csv"
if not manifest_path.is_file():
    st.error(f"Review manifest not found: {manifest_path}")
    st.stop()

st.set_page_config(page_title="Pothole Severity Review", layout="wide")
dataframe = _read_manifest(manifest_path)
label_column = (
    "adjudicated_label"
    if args.reviewer == "adjudicator"
    else f"{args.reviewer}_label"
)
notes_column = (
    "adjudication_notes"
    if args.reviewer == "adjudicator"
    else f"{args.reviewer}_notes"
)

if args.reviewer == "adjudicator":
    eligible = dataframe.index[
        (dataframe["reviewer_1_label"] != "")
        & (dataframe["reviewer_2_label"] != "")
        & (dataframe["reviewer_1_label"] != dataframe["reviewer_2_label"])
    ].tolist()
elif args.reviewer == "reviewer_2" and args.sample_percent < 100:
    if not 0 < args.sample_percent <= 100:
        st.error("--sample-percent must be above 0 and at most 100")
        st.stop()
    reviewed_by_first = dataframe[dataframe["reviewer_1_label"] != ""]
    sampled: list[int] = []
    for _, group in reviewed_by_first.groupby("reviewer_1_label", sort=True):
        sample_count = max(1, round(len(group) * args.sample_percent / 100))
        sampled.extend(
            group.sample(
                n=min(sample_count, len(group)), random_state=args.seed
            ).index.tolist()
        )
    eligible = sorted(sampled)
else:
    eligible = dataframe.index.tolist()

unreviewed = [index for index in eligible if not dataframe.at[index, label_column]]
if "review_index" not in st.session_state:
    st.session_state.review_index = (
        unreviewed[0] if unreviewed else (eligible[0] if eligible else 0)
    )

st.title("Pothole Severity Review")
reviewed_count = sum(bool(dataframe.at[index, label_column]) for index in eligible)
st.progress(reviewed_count / max(len(eligible), 1))
st.caption(
    f"Mode: {args.reviewer} · completed {reviewed_count}/{len(eligible)} · "
    f"manifest saves after every decision"
)

if not eligible:
    st.success("No items require review in this mode.")
    st.stop()

position = (
    eligible.index(st.session_state.review_index)
    if st.session_state.review_index in eligible
    else 0
)
selected_position = st.number_input(
    "Review item",
    min_value=1,
    max_value=len(eligible),
    value=position + 1,
    step=1,
)
index = eligible[int(selected_position) - 1]
st.session_state.review_index = index
row = dataframe.loc[index]

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Source image · {row['image_name']}")
    source_path = PROJECT_ROOT / row["image_path"]
    with Image.open(source_path) as source_image:
        highlighted = source_image.convert("RGB")
    draw = ImageDraw.Draw(highlighted)
    coordinates = tuple(int(float(row[name])) for name in ("x1", "y1", "x2", "y2"))
    line_width = max(4, round(min(highlighted.size) / 250))
    draw.rectangle(coordinates, outline=(255, 30, 30), width=line_width)
    draw.text((coordinates[0], max(0, coordinates[1] - 18)), row["detection_id"], fill=(255, 30, 30))
    st.image(highlighted, use_container_width=True)
with right:
    st.subheader(f"Crop · {row['detection_id']}")
    crop_path = PROJECT_ROOT / row["crop_path"]
    st.image(str(crop_path), use_container_width=True)
    st.write(f"Detector confidence: {float(row['confidence']):.3f}")
    if args.reviewer == "adjudicator":
        st.write(f"Reviewer 1: `{row['reviewer_1_label']}`")
        st.write(f"Reviewer 2: `{row['reviewer_2_label']}`")

notes = st.text_area("Notes", value=str(row[notes_column]), key=f"notes-{index}")
duplicate_of = st.text_input(
    "Duplicate of detection ID (only when applicable)",
    value=str(row["duplicate_of"]),
    key=f"duplicate-{index}",
)

columns = st.columns(3)
clicked_label = ""
for label_index, label in enumerate(REVIEW_LABELS):
    if columns[label_index % 3].button(
        label.replace("_", " ").title(), use_container_width=True
    ):
        clicked_label = label

if clicked_label:
    dataframe.at[index, label_column] = clicked_label
    dataframe.at[index, notes_column] = notes
    dataframe.at[index, "duplicate_of"] = duplicate_of
    _save(dataframe, manifest_path)
    remaining = [item for item in eligible if not dataframe.at[item, label_column]]
    st.session_state.review_index = remaining[0] if remaining else index
    st.rerun()

previous_column, next_column = st.columns(2)
if previous_column.button("Previous", use_container_width=True) and position > 0:
    st.session_state.review_index = eligible[position - 1]
    st.rerun()
if next_column.button("Next", use_container_width=True) and position < len(eligible) - 1:
    st.session_state.review_index = eligible[position + 1]
    st.rerun()

counts = dataframe[label_column].value_counts().reindex(REVIEW_LABELS, fill_value=0)
st.sidebar.subheader("Current labels")
st.sidebar.dataframe(counts.rename("count"), use_container_width=True)
