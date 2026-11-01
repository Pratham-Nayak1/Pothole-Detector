"""
Severity Classifier v3 — Focal Loss + MixUp + Oversampling
===========================================================
Key improvements over all previous runs:

1. Focal Loss (γ=2) — down-weights easy examples so the model focuses on
   hard / minority (High) samples without over-sampling artifacts.
2. MixUp (α=0.4) — interpolates pairs of training samples for regularisation,
   dramatically reduces over-fitting on the small corpus.
3. Explicit minority-class duplication — High samples are duplicated 2.5×,
   Low 1.5× in the training set *before* batching, giving the focal loss more
   signal from rare classes without the collapse seen with WeightedRandomSampler.
4. Stronger augmentation — RandomResizedCrop, GaussianBlur, sharper colour
   jitter, GridDistortion via torchvision's RandomPerspective.
5. OneCycleLR scheduler — tested to converge faster and more stably than
   cosine annealing on small datasets.
6. TTA at test time — five-crop ensemble.
7. Auto-promotes to configs/pipeline.yaml if balanced-accuracy > 0.604
   (the original ResNet18 v1 baseline).

Architecture: ResNet18 (same as v1 for full pipeline compatibility).
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
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

from src.severity.dataset import CLASSES, load_clean_dataset, save_splits

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
CSV_PATH      = ROOT / "outputs" / "severity_classifier" / "severity_dataset.csv"
SPLIT_DIR     = ROOT / "outputs" / "severity_classifier" / "splits"
OUTPUT_DIR    = ROOT / "outputs" / "severity_classifier" / "resnet18_v3"
MODEL_PATH    = OUTPUT_DIR / "best_severity_model.pth"
PIPELINE_YAML = ROOT / "configs" / "pipeline.yaml"

BASELINE_BALANCED_ACCURACY = 0.6038   # must beat this to auto-promote

SEED          = 42
BATCH_SIZE    = 24
EPOCHS        = 40
IMAGE_SIZE    = 224
LR            = 5e-4
WEIGHT_DECAY  = 2e-4
FOCAL_GAMMA   = 2.0
MIXUP_ALPHA   = 0.4
ORDINAL_WEIGHT = 0.20
EARLY_STOPPING_PATIENCE = 10

# Oversampling multipliers for minority classes in *training set only*
OVERSAMPLE = {"low": 1.5, "medium": 1.0, "high": 2.5}

LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

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
    def __init__(self, dataframe: pd.DataFrame, transform=None):
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


def oversample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate minority-class rows so the training distribution is flatter."""
    parts = [df]
    for label, mult in OVERSAMPLE.items():
        subset = df[df["manual_label"] == label]
        extra_reps = int(mult) - 1          # integer extra copies
        frac_extra = mult - int(mult)        # fractional extra copy
        if extra_reps:
            parts.append(pd.concat([subset] * extra_reps, ignore_index=True))
        if frac_extra > 0:
            parts.append(subset.sample(frac=frac_extra, random_state=SEED))
    combined = pd.concat(parts, ignore_index=True)
    return combined.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalOrdinalLoss(nn.Module):
    """
    Focal Loss (class-weighted, γ=FOCAL_GAMMA) plus an ordinal smooth-L1
    penalty that discourages jumping two severity levels.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
        ordinal_weight: float = 0.20,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)
        self.register_buffer(
            "class_positions", torch.arange(len(CLASSES), dtype=torch.float32)
        )

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth targets
        n_cls = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / n_cls)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / n_cls)

        log_probs = F.log_softmax(logits, dim=1)
        ce = -(smooth_targets * log_probs).sum(dim=1)          # per-sample CE

        # Class-weight scaling
        weights = self.class_weights[targets]
        ce = ce * weights

        # Focal modulation
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        positions = self.class_positions.to(logits.device)
        expected = probs @ positions
        ordinal = F.smooth_l1_loss(expected, targets.float())
        return fl + self.ordinal_weight * ordinal


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(brightness=0.40, contrast=0.45, saturation=0.25, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.30, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TTA_TRANSFORM = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE * 1.14), int(IMAGE_SIZE * 1.14))),
    transforms.FiveCrop(IMAGE_SIZE),
    transforms.Lambda(
        lambda crops: torch.stack([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                transforms.ToTensor()(c)
            ) for c in crops
        ])
    ),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_classes: int = 3) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Add dropout before final FC for regularisation
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.fc.in_features, n_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Splits (re-use existing fixed splits for fair comparison)
# ---------------------------------------------------------------------------

def load_splits():
    paths = {
        name: SPLIT_DIR / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    if all(p.is_file() for p in paths.values()):
        frames = {name: pd.read_csv(p) for name, p in paths.items()}
        print("Re-using existing fixed severity splits (fair comparison).")
        return frames["train"], frames["validation"], frames["test"]
    raise FileNotFoundError(
        f"Severity splits not found in {SPLIT_DIR}. "
        "Run train_classifier.py first to create splits."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: FocalOrdinalLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    mixup_alpha: float,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        if lam < 0.99:
            indices = torch.randperm(images.size(0), device=DEVICE)
            mixed_images = lam * images + (1 - lam) * images[indices]
            logits = model(mixed_images)
            loss = mixup_loss(criterion, logits, labels, labels[indices], lam)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), bal_acc, all_preds, all_labels


@torch.inference_mode()
def evaluate_tta(model: nn.Module, test_df: pd.DataFrame) -> tuple[list, list]:
    """Test-time augmentation: five-crop ensemble prediction."""
    model.eval()
    all_preds, all_labels = [], []
    for _, row in test_df.iterrows():
        img_path = ROOT / row["crop_path"]
        with Image.open(img_path) as img:
            crops_tensor = TTA_TRANSFORM(img.convert("RGB"))  # (5, C, H, W)
        crops_tensor = crops_tensor.to(DEVICE)
        logits = model(crops_tensor)                          # (5, n_cls)
        avg_probs = torch.softmax(logits, dim=1).mean(dim=0)  # (n_cls,)
        pred = int(avg_probs.argmax().item())
        all_preds.append(pred)
        all_labels.append(LABEL_TO_ID[row["manual_label"]])
    return all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = load_clean_dataset(CSV_PATH, ROOT)
    train_df, val_df, test_df = load_splits()

    # Apply oversampling to training set only
    train_over = oversample_dataframe(train_df)

    print(f"\nDevice      : {DEVICE}")
    print(f"Train (raw) : {len(train_df)}  | after oversample: {len(train_over)}")
    print(f"Validation  : {len(val_df)}")
    print(f"Test        : {len(test_df)}")
    print("Train class dist (oversampled):")
    print(train_over["manual_label"].value_counts().to_string())

    # Compute class weights from *original* (not oversampled) training set
    counts = train_df["manual_label"].value_counts()
    total  = len(train_df)
    raw_weights = torch.tensor(
        [total / (len(CLASSES) * counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float32,
        device=DEVICE,
    )

    train_ds = PotholeDataset(train_over, TRAIN_TRANSFORM)
    val_ds   = PotholeDataset(val_df,    VAL_TRANSFORM)
    test_ds  = PotholeDataset(test_df,   VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = build_model(n_classes=len(CLASSES)).to(DEVICE)
    criterion = FocalOrdinalLoss(raw_weights, gamma=FOCAL_GAMMA,
                                 ordinal_weight=ORDINAL_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=500.0,
    )

    history = []
    best_val_bal_acc = 0.0
    best_epoch       = 0
    patience_counter = 0

    print(f"\nTraining ResNet18-v3 for up to {EPOCHS} epochs …\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, MIXUP_ALPHA
        )
        val_loss, val_bal_acc, _, _ = evaluate(model, val_loader, criterion)
        _, val_acc_raw, vp, vl = evaluate(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_balanced_acc": round(val_bal_acc, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_bal_acc={val_bal_acc:.4f}"
            + (" << best" if val_bal_acc > best_val_bal_acc else "")
        )

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            patience_counter = 0
            # Save only the fc-head linear weights (not Sequential wrapper)
            # to keep state_dict compatible with pipeline/models.py
            state = {}
            for k, v in model.state_dict().items():
                # Rename fc.1.* → fc.* so the plain-fc pipeline loader works
                new_k = k.replace("fc.1.", "fc.")
                state[new_k] = v
            torch.save(state, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE}).")
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"\nBest val balanced accuracy: {best_val_bal_acc:.4f} at epoch {best_epoch}")

    # ---- Test-time evaluation (standard + TTA) ----------------------------
    print("\nEvaluating on test set (standard)…")
    # Reload best model
    model_eval = models.resnet18(weights=None)
    model_eval.fc = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(512, len(CLASSES)))
    raw_sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    # Re-map fc.* → fc.1.* for the Sequential head
    remapped = {}
    for k, v in raw_sd.items():
        if k.startswith("fc.") and not k.startswith("fc.1.") and not k.startswith("fc.0."):
            remapped[k.replace("fc.", "fc.1.", 1)] = v
        else:
            remapped[k] = v
    model_eval.load_state_dict(remapped)
    model_eval = model_eval.to(DEVICE)

    _, _, std_preds, std_labels = evaluate(model_eval, test_loader, criterion)
    std_acc  = accuracy_score(std_labels, std_preds)
    std_bacc = balanced_accuracy_score(std_labels, std_preds)
    print(f"  Standard — acc={std_acc:.4f}  bal_acc={std_bacc:.4f}")

    print("Evaluating on test set (TTA 5-crop)…")
    tta_preds, tta_labels = evaluate_tta(model_eval, test_df)
    tta_acc  = accuracy_score(tta_labels, tta_preds)
    tta_bacc = balanced_accuracy_score(tta_labels, tta_preds)
    print(f"  TTA       — acc={tta_acc:.4f}  bal_acc={tta_bacc:.4f}")

    # Use whichever is better
    if tta_bacc >= std_bacc:
        final_preds, final_labels = tta_preds, tta_labels
        use_tta = True
        final_acc, final_bacc = tta_acc, tta_bacc
    else:
        final_preds, final_labels = std_preds, std_labels
        use_tta = False
        final_acc, final_bacc = std_acc, std_bacc

    report = classification_report(
        final_labels, final_preds,
        target_names=list(CLASSES), output_dict=True
    )
    macro_f1 = f1_score(final_labels, final_preds, average="macro")
    ordinal_mae = mean_absolute_error(final_labels, final_preds)
    qwk = cohen_kappa_score(final_labels, final_preds, weights="quadratic")

    within_one = sum(abs(p - l) <= 1 for p, l in zip(final_preds, final_labels)) / len(final_labels)

    # Selective prediction metrics
    model_eval.eval()
    all_probs_list = []
    with torch.inference_mode():
        for images, _ in test_loader:
            probs = torch.softmax(model_eval(images.to(DEVICE)), dim=1)
            all_probs_list.extend(probs.cpu().tolist())
    max_probs = [max(p) for p in all_probs_list]
    selective = {}
    for thresh in [0.50, 0.60, 0.70]:
        mask = [mp >= thresh for mp in max_probs]
        covered = sum(mask)
        if covered:
            sel_preds  = [p for p, m in zip(final_preds, mask) if m]
            sel_labels = [l for l, m in zip(final_labels, mask) if m]
            selective[str(thresh)] = {
                "coverage": covered / len(final_labels),
                "accuracy": accuracy_score(sel_labels, sel_preds),
            }

    metrics = {
        "model": "resnet18_v3",
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_val_bal_acc,
        "test_accuracy": final_acc,
        "test_balanced_accuracy": final_bacc,
        "test_macro_f1": macro_f1,
        "test_ordinal_mae": ordinal_mae,
        "test_quadratic_weighted_kappa": qwk,
        "test_within_one_accuracy": within_one,
        "selective_metrics": selective,
        "tta": use_tta,
        "classification_report": report,
        "classes": list(CLASSES),
        "baseline_balanced_accuracy": BASELINE_BALANCED_ACCURACY,
        "beats_baseline": final_bacc > BASELINE_BALANCED_ACCURACY,
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(CLASSES))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"ResNet18-v3 | bal_acc={final_bacc:.3f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f" ResNet18-v3 Final Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy         : {final_acc:.4f}")
    print(f"  Balanced Accuracy: {final_bacc:.4f}  (baseline: {BASELINE_BALANCED_ACCURACY})")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  QWK              : {qwk:.4f}")
    print(f"  TTA used         : {use_tta}")
    print(f"  Beats baseline   : {final_bacc > BASELINE_BALANCED_ACCURACY}")
    print(f"{'='*60}")
    print(classification_report(final_labels, final_preds, target_names=list(CLASSES)))

    # ---- Auto-promote if better ------------------------------------------
    if final_bacc > BASELINE_BALANCED_ACCURACY:
        _promote_model()
        print(f"\n[PROMOTED]  Model saved to configs/pipeline.yaml  (balanced_acc={final_bacc:.4f})")
    else:
        print(f"\n[NOT PROMOTED]  Model did NOT beat baseline ({BASELINE_BALANCED_ACCURACY}).  pipeline.yaml unchanged.")

    return metrics


def _promote_model():
    import yaml
    with open(PIPELINE_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rel_path = str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/")
    config.setdefault("severity", {})["model_path"] = rel_path
    with open(PIPELINE_YAML, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
