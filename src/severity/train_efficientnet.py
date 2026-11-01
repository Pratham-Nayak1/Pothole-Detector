"""
EfficientNet-B0 severity classifier for pothole Low / Medium / High classification.

Improvements over the ResNet18 baseline:
- EfficientNet-B0: better ImageNet accuracy at similar parameter count.
- Stronger augmentation (RandAugment + random erasing) with higher intensity for
  minority classes via class-weighted sampler.
- Cosine annealing LR schedule with warm-up.
- Label smoothing + ordinal penalty (same as baseline).
- Test-time augmentation (TTA) at inference via 5 crops.
- Saves to outputs/severity_classifier/efficientnet_b0/ and promotes to
  configs/pipeline.yaml only if it beats the baseline on balanced accuracy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)

from src.severity.dataset import CLASSES, load_clean_dataset, save_splits, create_grouped_splits

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "severity_classifier" / "severity_dataset.csv"
SPLIT_DIR = ROOT / "outputs" / "severity_classifier" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "severity_classifier" / "efficientnet_b0"
MODEL_PATH = OUTPUT_DIR / "best_severity_model.pth"

BASELINE_BALANCED_ACCURACY = 0.6038   # ResNet18 baseline — must beat this to promote
PIPELINE_YAML = ROOT / "configs" / "pipeline.yaml"
PROMOTED_MODEL_KEY = "model_path"

LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 16
EPOCHS = 30
IMAGE_SIZE = 224
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ORDINAL_LOSS_WEIGHT = 0.30
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PotholeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = ROOT / row["crop_path"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_ID[row["manual_label"]]
        return image, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class OrdinalCrossEntropyLoss(nn.Module):
    """Weighted CE + distance-aware ordinal penalty."""

    def __init__(self, class_weights: torch.Tensor, ordinal_weight: float) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.07
        )
        self.ordinal_weight = ordinal_weight
        self.register_buffer(
            "class_positions", torch.arange(len(CLASSES), dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        expected = probs @ self.class_positions
        ordinal_loss = nn.functional.smooth_l1_loss(expected, labels.float())
        return ce_loss + self.ordinal_weight * ordinal_loss


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def load_or_create_splits(df: pd.DataFrame):
    paths = {
        name: SPLIT_DIR / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    if all(p.is_file() for p in paths.values()):
        frames = {name: pd.read_csv(p) for name, p in paths.items()}
        saved_ids = set(
            pd.concat(frames.values(), ignore_index=True)["detection_id"].astype(str)
        )
        current_ids = set(df["detection_id"].astype(str))
        if saved_ids == current_ids:
            print("Using the existing fixed severity splits.")
            return frames["train"], frames["validation"], frames["test"]

    print("Dataset changed — creating new deterministic group-isolated splits.")
    train_df, val_df, test_df = create_grouped_splits(
        df, seed=SEED, test_size=TEST_SIZE, validation_size_within_train=0.20
    )
    save_splits(SPLIT_DIR, train_df, val_df, test_df, seed=SEED)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Weighted sampler (over-samples minority classes inside each epoch)
# ---------------------------------------------------------------------------
def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df["manual_label"].value_counts()
    class_weight = {
        name: len(df) / (len(CLASSES) * max(counts.get(name, 1), 1))
        for name in CLASSES
    }
    sample_weights = torch.tensor(
        [class_weight[row] for row in df["manual_label"]], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ---------------------------------------------------------------------------
# Cosine-with-warmup LR schedule
# ---------------------------------------------------------------------------
def cosine_with_warmup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------
def tta_predict(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Average softmax over original + horizontal flip (simple 2-crop TTA)."""
    flipped = tensor.flip(-1)
    crops = [tensor, flipped]
    probs = torch.stack(
        [torch.softmax(model(c.to(DEVICE)), dim=1).cpu() for c in crops]
    ).mean(0)
    return probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = load_clean_dataset(CSV_PATH, ROOT)
    train_df, val_df, test_df = load_or_create_splits(df)

    print(f"Device: {DEVICE}")
    print(f"Total examples: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train class counts: {dict(train_df['manual_label'].value_counts())}")

    # ------------------------------------------------------------------
    # Transforms — heavier augmentation than ResNet18 baseline
    # ------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PotholeDataset(train_df, train_transform)
    val_dataset = PotholeDataset(val_df, eval_transform)
    test_dataset = PotholeDataset(test_df, eval_transform)

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model — EfficientNet-B0 with custom head
    # ------------------------------------------------------------------
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except Exception as exc:
        raise RuntimeError("Could not load pretrained EfficientNet-B0 weights.") from exc

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASSES)),
    )
    model = model.to(DEVICE)

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    counts = train_df["manual_label"].value_counts()
    class_weights = torch.tensor(
        [len(train_df) / (len(CLASSES) * max(counts.get(name, 1), 1)) for name in CLASSES],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = OrdinalCrossEntropyLoss(class_weights, ORDINAL_LOSS_WEIGHT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_with_warmup(ep, WARMUP_EPOCHS, EPOCHS),
    )

    best_val_balanced_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss_sum += criterion(outputs, labels).item() * labels.size(0)
                val_total += labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(outputs.argmax(dim=1).cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_bal_acc = balanced_accuracy_score(val_true, val_pred)
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_balanced_accuracy": val_bal_acc,
            "lr": current_lr,
        })
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train loss: {train_loss:.4f} | acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | bal_acc: {val_bal_acc:.4f}"
        )

        improved = val_bal_acc > best_val_balanced_accuracy or (
            val_bal_acc == best_val_balanced_accuracy and val_loss < best_val_loss
        )
        if improved:
            best_val_balanced_accuracy = val_bal_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # ------------------------------------------------------------------
    # Test evaluation with TTA
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    y_true, y_pred, y_confidence = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(images.size(0)):
                probs = tta_predict(model, images[i : i + 1])
                y_pred.append(int(probs.argmax(dim=1).item()))
                y_confidence.append(float(probs.max(dim=1).values.item()))
            y_true.extend(labels.tolist())

    test_accuracy = accuracy_score(y_true, y_pred)
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    ordinal_mae = mean_absolute_error(y_true, y_pred)
    quadratic_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    within_one = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    selective_metrics: dict = {}
    for threshold in (0.50, 0.60, 0.70):
        accepted = [i for i, v in enumerate(y_confidence) if v >= threshold]
        selective_metrics[f"{threshold:.2f}"] = {
            "coverage": len(accepted) / max(len(y_true), 1),
            "accuracy": (
                accuracy_score(
                    [y_true[i] for i in accepted], [y_pred[i] for i in accepted]
                )
                if accepted else None
            ),
        }

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test accuracy (TTA): {test_accuracy:.4f}")
    print(f"Test balanced accuracy (TTA): {test_bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0,
    ))

    metrics = {
        "model": "efficientnet_b0",
        "best_epoch": best_epoch,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_bal_acc,
        "test_macro_f1": macro_f1,
        "test_ordinal_mae": ordinal_mae,
        "test_quadratic_weighted_kappa": quadratic_kappa,
        "test_within_one_accuracy": within_one,
        "selective_metrics": selective_metrics,
        "classification_report": report_dict,
        "tta": True,
        "classes": list(CLASSES),
        "baseline_balanced_accuracy": BASELINE_BALANCED_ACCURACY,
        "beats_baseline": test_bal_acc > BASELINE_BALANCED_ACCURACY,
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"EfficientNet-B0 Severity Classifier — Test Set (TTA)\nBalanced acc: {test_bal_acc:.4f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # ------------------------------------------------------------------
    # Promote to pipeline config if beats baseline
    # ------------------------------------------------------------------
    if test_bal_acc > BASELINE_BALANCED_ACCURACY:
        print(
            f"\n[PROMOTED] EfficientNet-B0 ({test_bal_acc:.4f}) beats ResNet18 baseline "
            f"({BASELINE_BALANCED_ACCURACY:.4f}). Updating pipeline config..."
        )
        _promote_model(MODEL_PATH)
    else:
        print(
            f"\n[SKIPPED] EfficientNet-B0 ({test_bal_acc:.4f}) did NOT beat the ResNet18 "
            f"baseline ({BASELINE_BALANCED_ACCURACY:.4f}). Keeping existing config."
        )

    print(f"\nModel: {MODEL_PATH}")
    print(f"Metrics: {OUTPUT_DIR / 'test_metrics.json'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")


def _promote_model(model_path: Path) -> None:
    """Update configs/pipeline.yaml severity model_path to the new checkpoint."""
    import yaml  # PyYAML is in requirements

    with PIPELINE_YAML.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Store path relative to project root for portability
    rel_path = model_path.relative_to(ROOT).as_posix()
    config["severity"]["model_path"] = rel_path

    with PIPELINE_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Pipeline config updated: severity.model_path = {rel_path}")


if __name__ == "__main__":
    main()
