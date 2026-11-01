
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "configs" / "data.yaml"

# Start from our completed YOLO11s model
MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolo11s_640_baseline"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "training"

EPOCHS = 30
IMAGE_SIZE = 800
BATCH_SIZE = 8
DEVICE = 0


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {MODEL_PATH}")

    print("Starting YOLO11s fine-tuning at 800 pixels")
    print(f"Starting weights: {MODEL_PATH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="yolo11s_800_finetune",
        exist_ok=False,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        patience=10,
    )

    print("Training finished.")
    print(
        "Results directory:",
        OUTPUT_DIR / "yolo11s_800_finetune",
    )


if __name__ == "__main__":
    main()
