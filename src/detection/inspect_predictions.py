
from pathlib import Path
import random
import csv

from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "images" / "val"

MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolo11s_640_baseline"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "inspection" / "yolo11s_random20"

# --------------------------------------------------
# 2. SETTINGS - CHANGE THESE IF REQUIRED
# --------------------------------------------------
NUM_IMAGES = 20
CONFIDENCE = 0.10       # Show detections with confidence >= 10%
IMAGE_SIZE = 640
DEVICE = 0              # NVIDIA GPU
RANDOM_SEED = 42        # Reproducible random selection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Check that the paths exist.
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # Find all validation images.
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) < NUM_IMAGES:
        raise ValueError(
            f"Need {NUM_IMAGES} images, but found only {len(image_paths)}."
        )

    # Randomly select exactly 20 images.
    rng = random.Random(RANDOM_SEED)
    selected_images = rng.sample(image_paths, NUM_IMAGES)

    # Create a fresh output folder so old results aren't mixed in.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL_PATH}")
    print(f"Available images: {len(image_paths)}")
    print(f"Selected images: {len(selected_images)}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"Output folder: {OUTPUT_DIR}")

    model = YOLO(str(MODEL_PATH))
    report_rows = []

    # Process each selected image separately.
    for index, image_path in enumerate(selected_images, start=1):
        print(f"\n[{index}/{NUM_IMAGES}] {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            save=False,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        detection_count = len(boxes) if boxes is not None else 0

        # Save an annotated image even when there are no detections.
        annotated_image = result.plot()
        output_image = OUTPUT_DIR / f"{index:02d}_{image_path.name}"

        import cv2
        if not cv2.imwrite(str(output_image), annotated_image):
            raise OSError(f"Could not save image: {output_image}")

        print(f"Detections: {detection_count}")
        print(f"Saved: {output_image.name}")

        # Record all detections in a CSV report.
        if detection_count:
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = model.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                report_rows.append({
                    "image": image_path.name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                })
        else:
            # Preserve images with no detections in the report.
            report_rows.append({
                "image": image_path.name,
                "class_id": "",
                "class_name": "NO DETECTIONS",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            })

    # Save a CSV report of detections and empty results.
    report_path = OUTPUT_DIR / "detection_report.csv"
    fieldnames = [
        "image", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2"
    ]

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nInspection complete.")
    print(f"Images processed: {len(selected_images)}")
    print(f"Report saved: {report_path}")
    print(f"Open this folder to inspect the images:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
