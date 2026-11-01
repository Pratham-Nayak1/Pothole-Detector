
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"

MODEL_NAME = "yolo11s.pt"
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 12
DEVICE = 0


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    print("Starting YOLO11s road-defect training")
    print(f"Dataset: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"GPU device: {DEVICE}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="yolo11s_640_baseline",
        exist_ok=False,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        patience=10,
    )

    print("Training finished.")
    print(
        "Results:",
        OUTPUT_DIR / "yolo11s_640_baseline",
    )


if __name__ == "__main__":
    main()
