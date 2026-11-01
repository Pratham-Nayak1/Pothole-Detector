
from pathlib import Path
from ultralytics import YOLO

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"

MODEL_NAME = "yolo11n.pt"
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0  # NVIDIA GPU

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    print("Starting fresh road-defect training")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Model initialization: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="yolo11n_baseline",
        exist_ok=False,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
    )

    print("Training finished.")
    print(f"Results directory: {OUTPUT_DIR / 'yolo11n_baseline'}")

if __name__ == "__main__":
    main()
