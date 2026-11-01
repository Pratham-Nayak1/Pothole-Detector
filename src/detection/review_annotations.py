
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "images" / "val"
LABEL_DIR = ROOT / "data" / "labels" / "val"

MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolo11s_640_baseline" / "weights" / "best.pt"
)

SAMPLE_DIR = ROOT / "outputs" / "inspection" / "yolo11s_random20"
SAMPLE_REPORT = SAMPLE_DIR / "detection_report.csv"

OUTPUT_DIR = ROOT / "outputs" / "annotation_review"

CONFIDENCE = 0.10
IMAGE_SIZE = 640
NMS_IOU = 0.70  # Match the previous prediction setting
DEVICE = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_ground_truth(image, label_path, model_names):
    """Draw ground-truth boxes from YOLO-format label files."""
    image = image.copy()

    if not label_path.exists():
        cv2.putText(
            image, "LABEL FILE MISSING",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )
        return image, None

    lines = [
        line.split()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    height, width = image.shape[:2]
    valid_count = 0

    for row in lines:
        if len(row) < 5:
            continue

        try:
            class_id = int(row[0])
            xc, yc, bw, bh = map(float, row[1:5])
        except ValueError:
            continue

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        class_name = model_names.get(class_id, str(class_id))

        # Green boxes = dataset ground truth
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            image, class_name, (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2
        )
        valid_count += 1

    return image, valid_count


def main():
    for path in (IMAGE_DIR, LABEL_DIR, MODEL_PATH, SAMPLE_REPORT):
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    with SAMPLE_REPORT.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        sample_rows = list(csv.DictReader(file))

    # Recover the unique image names from the earlier random sample.
    image_names = sorted({
        row["image"] for row in sample_rows if row.get("image")
    })

    if len(image_names) != 20:
        raise ValueError(
            f"Expected 20 sampled images, found {len(image_names)}. "
            "Check detection_report.csv."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    report = []

    print(f"Reviewing {len(image_names)} images")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"Output folder: {OUTPUT_DIR}")

    for index, image_name in enumerate(image_names, start=1):
        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / f"{Path(image_name).stem}.txt"

        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Skipping unreadable image: {image_name}")
            continue

        gt_image, gt_count = draw_ground_truth(
            original, label_path, model.names
        )

        result = model.predict(
            source=original,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device=DEVICE,
            save=False,
            verbose=False,
        )[0]

        prediction_image = result.plot()
        prediction_count = (
            len(result.boxes) if result.boxes is not None else 0
        )

        # Label each panel.
        cv2.putText(
            gt_image, "GROUND TRUTH",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )
        cv2.putText(
            prediction_image, "YOLO11s PREDICTIONS",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )

        # Keep the panels the same size before joining them.
        if gt_image.shape[0] != prediction_image.shape[0]:
            prediction_image = cv2.resize(
                prediction_image,
                (gt_image.shape[1], gt_image.shape[0])
            )

        comparison = np.hstack((gt_image, prediction_image))

        output_path = OUTPUT_DIR / f"{index:02d}_comparison.jpg"
        if not cv2.imwrite(str(output_path), comparison):
            raise OSError(f"Could not save {output_path}")

        report.append({
            "image": image_name,
            "label_file_exists": label_path.exists(),
            "ground_truth_boxes": (
                gt_count if gt_count is not None else "MISSING LABEL FILE"
            ),
            "predicted_boxes": prediction_count,
            "comparison_image": output_path.name,
        })

        print(
            f"[{index:02d}/20] {image_name}: "
            f"GT={gt_count}, predictions={prediction_count}"
        )

    report_path = OUTPUT_DIR / "annotation_review.csv"
    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    print("\nReview images saved.")
    print(f"Folder: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
