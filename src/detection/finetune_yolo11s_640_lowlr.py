"""Conservative AdamW fine-tuning from the strongest 640 px checkpoint.

The previous experiment requested a low learning rate while leaving the optimizer
on ``auto``. Ultralytics selected a much larger effective rate, so that run did
not test the intended hypothesis. This recipe pins the optimizer and disables
warm-up to avoid that failure mode.
"""

from pathlib import Path

from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "configs" / "data.yaml"

MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "yolo11s_640_baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "training"

# Training configuration
EPOCHS = 15
IMAGE_SIZE = 640
BATCH_SIZE = 12
DEVICE = 0

# Lower learning rate for fine-tuning
LEARNING_RATE = 0.0001
FINAL_LR_FACTOR = 0.10

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {MODEL_PATH}"
        )

    print("=" * 55)
    print("YOLO11s 640px - AdamW Fine-tuning")
    print("=" * 55)
    print(f"Starting weights: {MODEL_PATH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Pin the optimizer: optimizer="auto" ignores a user-provided lr0.
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        lrf=FINAL_LR_FACTOR,
        cos_lr=True,
        warmup_epochs=0,
        weight_decay=0.0005,

        # Reproducibility and checkpoint safety
        seed=42,
        deterministic=True,
        patience=5,
        save=True,
        plots=True,

        # New output directory; preserve the baseline
        project=str(OUTPUT_DIR),
        name="yolo11s_640_adamw_finetune",
        exist_ok=False,
    )

    print("\nTraining finished.")
    print(
        "Results directory:",
        OUTPUT_DIR / "yolo11s_640_adamw_finetune"
    )


if __name__ == "__main__":
    main()
