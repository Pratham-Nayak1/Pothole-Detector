
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "configs" / "data.yaml"

MODELS = {
    "YOLO11n baseline": (
        ROOT / "outputs" / "training" / "yolo11n_baseline"
        / "weights" / "best.pt"
    ),
    "YOLO11s 640 batch12": (
        ROOT / "outputs" / "training" / "yolo11s_640_baseline"
        / "weights" / "best.pt"
    ),
}

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config missing: {DATA}")

    for name, weights in MODELS.items():
        if not weights.exists():
            raise FileNotFoundError(f"Checkpoint missing: {weights}")

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"Weights: {weights}")
        print("=" * 60)

        model = YOLO(str(weights))
        model.val(
            data=str(DATA),
            split="val",
            imgsz=640,
            batch=8,
            device=0,
            plots=True,
            project=str(ROOT / "outputs" / "comparison"),
            name=name.lower().replace(" ", "_"),
            exist_ok=True,
        )

if __name__ == "__main__":
    main()
