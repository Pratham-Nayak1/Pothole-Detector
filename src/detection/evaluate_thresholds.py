
from pathlib import Path
import csv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolo11s_640_baseline" / "weights" / "best.pt"
)
DATA_YAML = ROOT / "configs" / "data.yaml"

OUTPUT_DIR = ROOT / "outputs" / "threshold_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold combinations to compare
CONFIDENCE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
NMS_IOU_THRESHOLDS = [0.40, 0.50, 0.60, 0.70]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))
    results = []

    for conf in CONFIDENCE_THRESHOLDS:
        for iou in NMS_IOU_THRESHOLDS:
            print(f"\nEvaluating conf={conf}, NMS IoU={iou}")

            metrics = model.val(
                data=str(DATA_YAML),
                imgsz=640,
                batch=8,
                device=0,
                conf=conf,
                iou=iou,
                plots=False,
                verbose=False,
                project=str(OUTPUT_DIR),
                name=f"conf_{conf}_iou_{iou}",
                exist_ok=True,
            )

            row = {
                "confidence": conf,
                "nms_iou": iou,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            }
            results.append(row)

            print(
                f"P={row['precision']:.4f} | "
                f"R={row['recall']:.4f} | "
                f"mAP50={row['mAP50']:.4f} | "
                f"mAP50-95={row['mAP50_95']:.4f}"
            )

    csv_path = OUTPUT_DIR / "threshold_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results saved to: {csv_path}")
    print("\nTop configurations by mAP50:")
    for row in sorted(
        results, key=lambda x: x["mAP50"], reverse=True
    )[:5]:
        print(row)

if __name__ == "__main__":
    main()
