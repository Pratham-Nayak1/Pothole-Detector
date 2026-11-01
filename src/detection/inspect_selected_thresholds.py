
from pathlib import Path
import random
import csv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolo11s_640_baseline" / "weights" / "best.pt"
)
VAL_IMAGES = ROOT / "data" / "images" / "val"
OUTPUT_DIR = ROOT / "outputs" / "threshold_test"

CONFIDENCE = 0.20
NMS_IOU = 0.40
SAMPLE_SIZE = 30
SEED = 42

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    images = sorted(
        p for p in VAL_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not images:
        raise FileNotFoundError(f"No validation images found in {VAL_IMAGES}")

    random.seed(SEED)
    selected = random.sample(images, min(SAMPLE_SIZE, len(images)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    rows = []

    for image_path in selected:
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device=0,
            verbose=False,
        )[0]

        output_path = OUTPUT_DIR / image_path.name
        result.save(filename=str(output_path))

        boxes = result.boxes
        rows.append({
            "image": image_path.name,
            "detections": len(boxes),
            "output_image": str(output_path),
        })

        print(f"{image_path.name}: {len(boxes)} detections")

    report_path = OUTPUT_DIR / "detection_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAnnotated images: {OUTPUT_DIR}")
    print(f"Detection report: {report_path}")

if __name__ == "__main__":
    main()
