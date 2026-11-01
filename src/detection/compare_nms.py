
from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "images" / "val"
SAMPLE_REPORT = (
    ROOT / "outputs" / "inspection" / "yolo11s_random20"
    / "detection_report.csv"
)
MODEL_PATH = (
    ROOT / "outputs" / "training" / "yolo11s_640_baseline"
    / "weights" / "best.pt"
)
OUTPUT_DIR = ROOT / "outputs" / "inspection" / "nms_comparison"

CONFIDENCE = 0.10
IMAGE_SIZE = 640
NMS_THRESHOLDS = [0.70, 0.50, 0.40]
DEVICE = 0


def main():
    with SAMPLE_REPORT.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    image_names = sorted({row["image"] for row in rows if row.get("image")})
    if len(image_names) != 20:
        raise ValueError(f"Expected 20 images; found {len(image_names)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    summary = []

    for name in image_names:
        path = IMAGE_DIR / name
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {name}")
            continue

        panels = [image.copy()]
        counts = []

        for threshold in NMS_THRESHOLDS:
            result = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=threshold,
                device=DEVICE,
                save=False,
                verbose=False,
            )[0]

            count = len(result.boxes) if result.boxes is not None else 0
            counts.append(count)

            panel = result.plot()
            cv2.putText(
                panel,
                f"NMS IoU={threshold:.2f} | boxes={count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            panels.append(panel)

        height, width = panels[0].shape[:2]
        panels = [
            cv2.resize(p, (width, height)) for p in panels
        ]
        comparison = np.hstack(panels)

        output_path = OUTPUT_DIR / f"{Path(name).stem}_nms.jpg"
        if not cv2.imwrite(str(output_path), comparison):
            raise OSError(f"Could not save {output_path}")

        summary.append({
            "image": name,
            "boxes_iou_070": counts[0],
            "boxes_iou_050": counts[1],
            "boxes_iou_040": counts[2],
            "comparison_image": output_path.name,
        })

        print(
            f"{name}: NMS 0.70={counts[0]}, "
            f"0.50={counts[1]}, 0.40={counts[2]}"
        )

    with (OUTPUT_DIR / "nms_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved comparisons to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
