
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

from src.severity.dataset import (
    CLASSES,
    create_grouped_splits,
    load_clean_dataset,
    save_splits,
)

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "severity_classifier" / "severity_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "severity_classifier" / "resnet18_v2"
MODEL_PATH = OUTPUT_DIR / "best_severity_model.pth"
PIPELINE_YAML = ROOT / "configs" / "pipeline.yaml"
BASELINE_BALANCED_ACCURACY = 0.6038   # v1 ResNet18 on 310-sample dataset

LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 16
EPOCHS = 25
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
ORDINAL_LOSS_WEIGHT = 0.25
EARLY_STOPPING_PATIENCE = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PotholeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = ROOT / row["crop_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = LABEL_TO_ID[row["manual_label"]]
        return image, label


class OrdinalCrossEntropyLoss(nn.Module):
    """Weighted classification loss plus distance-aware severity penalty."""

    def __init__(self, class_weights: torch.Tensor, ordinal_weight: float) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.05
        )
        self.ordinal_weight = ordinal_weight
        self.register_buffer(
            "class_positions", torch.arange(len(CLASSES), dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classification_loss = self.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        expected_severity = probabilities @ self.class_positions
        ordinal_loss = nn.functional.smooth_l1_loss(
            expected_severity, labels.float()
        )
        return classification_loss + self.ordinal_weight * ordinal_loss


def load_or_create_splits(df: pd.DataFrame):
    split_dir = ROOT / "outputs" / "severity_classifier" / "splits"
    paths = {
        name: split_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    if all(path.is_file() for path in paths.values()):
        frames = {name: pd.read_csv(path) for name, path in paths.items()}
        saved_ids = set(
            pd.concat(frames.values(), ignore_index=True)["detection_id"].astype(str)
        )
        current_ids = set(df["detection_id"].astype(str))
        if saved_ids == current_ids:
            print("Using the existing fixed severity splits.")
            return frames["train"], frames["validation"], frames["test"]

    print("Dataset changed; creating new deterministic group-isolated splits.")
    train_df, val_df, test_df = create_grouped_splits(
        df,
        seed=SEED,
        test_size=TEST_SIZE,
        validation_size_within_train=0.20,
    )
    save_splits(split_dir, train_df, val_df, test_df, seed=SEED)
    return train_df, val_df, test_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = load_clean_dataset(CSV_PATH, ROOT)
    train_df, val_df, test_df = load_or_create_splits(df)

    print(f"Device: {DEVICE}")
    print(f"Total examples: {len(df)}")
    print(
        f"Train: {len(train_df)} examples from "
        f"{train_df['image_name'].nunique()} images"
    )
    print(
        f"Validation: {len(val_df)} examples from "
        f"{val_df['image_name'].nunique()} images"
    )
    print(
        f"Test: {len(test_df)} examples from "
        f"{test_df['image_name'].nunique()} images"
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.08
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = PotholeDataset(train_df, train_transform)
    val_dataset = PotholeDataset(val_df, eval_transform)
    test_dataset = PotholeDataset(test_df, eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ImageNet-pretrained ResNet18, adapted to our three severity classes.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ResNet18 weights. "
            "Check your internet connection and retry."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    # Class-weighted loss helps account for fewer high-severity examples.
    counts = train_df["manual_label"].value_counts()
    class_weights = torch.tensor(
        [
            len(train_df) / (len(CLASSES) * counts.get(name, 1))
            for name in CLASSES
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = OrdinalCrossEntropyLoss(
        class_weights, ORDINAL_LOSS_WEIGHT
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_val_balanced_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_true = []
        val_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_total += labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(outputs.argmax(dim=1).cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_balanced_accuracy = balanced_accuracy_score(val_true, val_pred)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_balanced_accuracy": val_balanced_accuracy,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val balanced accuracy: {val_balanced_accuracy:.4f}"
        )

        improved = (
            val_balanced_accuracy > best_val_balanced_accuracy
            or (
                val_balanced_accuracy == best_val_balanced_accuracy
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_val_loss = val_loss
            best_val_balanced_accuracy = val_balanced_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation score did not improve.")
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Evaluate the selected checkpoint on the held-out test images.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []
    y_confidence = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
            y_confidence.extend(probabilities.max(dim=1).values.cpu().tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation balanced accuracy: {best_val_balanced_accuracy:.4f}")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Test balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )
    print("\nTest classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    ordinal_mae = mean_absolute_error(y_true, y_pred)
    quadratic_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    within_one_accuracy = float(
        np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)
    )
    selective_metrics = {}
    for threshold in (0.50, 0.60, 0.70):
        accepted = [index for index, value in enumerate(y_confidence) if value >= threshold]
        selective_metrics[f"{threshold:.2f}"] = {
            "coverage": len(accepted) / max(len(y_true), 1),
            "accuracy": (
                accuracy_score(
                    [y_true[index] for index in accepted],
                    [y_pred[index] for index in accepted],
                )
                if accepted
                else None
            ),
        }

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "best_validation_balanced_accuracy": best_val_balanced_accuracy,
                "test_accuracy": accuracy_score(y_true, y_pred),
                "test_balanced_accuracy": balanced_accuracy_score(
                    y_true, y_pred
                ),
                "test_macro_f1": macro_f1,
                "test_ordinal_mae": ordinal_mae,
                "test_quadratic_weighted_kappa": quadratic_kappa,
                "test_within_one_accuracy": within_one_accuracy,
                "selective_metrics": selective_metrics,
                "classification_report": report,
                "classes": CLASSES,
                "ordinal_loss_weight": ORDINAL_LOSS_WEIGHT,
            },
            f,
            indent=2,
        )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASSES
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Pothole Severity Classifier — Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {OUTPUT_DIR / 'test_metrics.json'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")

    # Promote to pipeline.yaml if it beats the deployed baseline
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    if bal_acc > BASELINE_BALANCED_ACCURACY:
        print(
            f"\n[PROMOTED] ResNet18-v2 ({bal_acc:.4f}) beats baseline "
            f"({BASELINE_BALANCED_ACCURACY:.4f}). Updating pipeline config..."
        )
        _promote_model(MODEL_PATH)
    else:
        print(
            f"\n[SKIPPED] ResNet18-v2 ({bal_acc:.4f}) did not beat baseline "
            f"({BASELINE_BALANCED_ACCURACY:.4f}). Keeping existing config."
        )


def _promote_model(model_path: Path) -> None:
    import yaml
    with PIPELINE_YAML.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rel = model_path.relative_to(ROOT).as_posix()
    config["severity"]["model_path"] = rel
    with PIPELINE_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Pipeline config updated: severity.model_path = {rel}")


if __name__ == "__main__":
    main()
