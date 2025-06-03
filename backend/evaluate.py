from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Sequence

import h5py  # only needed indirectly, but ensures requirement check early
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import timm

# -----------------------------------------------------------------------------
# IMPORT dataset + constants FROM backend_old_model.py
# -----------------------------------------------------------------------------
# Assumption: evaluate.py and backend_old_model.py live in the same folder.
# backend_old_model.py defines Galaxy10H5, IMG_SIZE, MEAN, STD, H5_PATH.
# They are used for the *old* model and to give us a handy dataset wrapper.
# -----------------------------------------------------------------------------
from backend_old_model import Galaxy10H5, IMG_SIZE, MEAN, STD, H5_PATH

# -----------------------------------------------------------------------------
# Class names (the identical order for all 3 evaluators)
# -----------------------------------------------------------------------------
_CLASS_NAMES: List[str] = [
    "Smooth",
    "Features/Disk",
    "Star",
    "Edge-On Disk",
    "Spiral",
    "Irregular",
    "Lens/Arc",
    "Ring",
    "Disturbed",
    "Merger",
]
NUM_CLASSES = len(_CLASS_NAMES)

# -----------------------------------------------------------------------------
# Model builders – old (B0) & new (B2)
# -----------------------------------------------------------------------------

def _fix_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return {k.replace("model.", ""): v for k, v in raw.items()}


def build_old(weights: str | Path, device: torch.device) -> torch.nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    m.to(device)
    ckpt = torch.load(weights, map_location=device)
    m.load_state_dict(_fix_state_dict(ckpt), strict=False)
    m.eval()
    return m


def build_new(weights: str | Path, device: torch.device) -> torch.nn.Module:
    m = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(weights, map_location=device)
    m.load_state_dict(_fix_state_dict(ckpt), strict=False)
    m.eval()
    return m

# -----------------------------------------------------------------------------
# Transforms for each model (must mirror training!)
# -----------------------------------------------------------------------------
_TRANSFORM_OLD = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_TRANSFORM_NEW = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((260, 260), antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def _evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    conf = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    acc = accuracy_score(y_true, y_pred)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "confusion_matrix": conf,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for cls, p, r, f in zip(_CLASS_NAMES, per_p, per_r, per_f1)
        },
    }


def _print_cm(cm: np.ndarray):
    cell_w = max(len(c) for c in _CLASS_NAMES) + 2
    header = "".ljust(cell_w) + "".join(c.ljust(cell_w) for c in _CLASS_NAMES)
    print(header)
    for cls, row in zip(_CLASS_NAMES, cm):
        print(cls.ljust(cell_w) + "".join(str(v).ljust(cell_w) for v in row))


def _print_metrics(title: str, res: dict):
    print(f"\n===== {title} =====")
    _print_cm(res["confusion_matrix"])
    print("\nPer‑Class F1‑Scores:")
    for cls, m in res["per_class"].items():
        print(
            f"  {cls:>16}: F1={m['f1']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}"
        )
    print(f"\nTop‑1 Accuracy: {res['accuracy']*100:.2f}%  |  Macro‑F1: {res['macro_f1']:.4f}")

# -----------------------------------------------------------------------------
# Prediction helpers (return label id + max‑prob for each sample)
# -----------------------------------------------------------------------------

def _get_preds(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    preds, confs = [], []
    with torch.no_grad():
        for x, _ in loader:  # loader yields (tensor, label)
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, 1)
            c, p = torch.max(prob, 1)
            preds.append(p.cpu().numpy())
            confs.append(c.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    P = argparse.ArgumentParser(description="Evaluate old/new Galaxy10 models & ensemble")
    P.add_argument("--h5_path", type=str, default=str(H5_PATH), help="Galaxy10_DECals.h5")
    P.add_argument("--old_ckpt", type=str, default="galaxy10_b0.pth", help="EffNet‑B0 checkpoint")
    P.add_argument("--new_ckpt", type=str, default="galaxy10_ema.pth", help="EffNet‑B2 checkpoint")
    P.add_argument("--batch", type=int, default=64, help="Batch size for evaluation")
    return P.parse_args(argv)

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    cfg = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # ------------------- Load dataset once -------------------
    full_ds_raw = Galaxy10H5(cfg.h5_path, transform=None)  # transform applied later
    N = len(full_ds_raw)
    n_val, n_test = int(0.10 * N), int(0.10 * N)
    n_train = N - n_val - n_test
    g = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(full_ds_raw, [n_train, n_val, n_test], generator=g)
    idx_val, idx_test = val_subset.indices, test_subset.indices

    # ------------------- Datasets & loaders per model -------------------
    val_old = Galaxy10H5(cfg.h5_path, indices=idx_val, transform=_TRANSFORM_OLD)
    test_old = Galaxy10H5(cfg.h5_path, indices=idx_test, transform=_TRANSFORM_OLD)
    val_new = Galaxy10H5(cfg.h5_path, indices=idx_val, transform=_TRANSFORM_NEW)
    test_new = Galaxy10H5(cfg.h5_path, indices=idx_test, transform=_TRANSFORM_NEW)

    NUM_WORKERS = 0  # h5py on Windows
    val_loader_old = DataLoader(val_old, batch_size=cfg.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_old = DataLoader(test_old, batch_size=cfg.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_new = DataLoader(val_new, batch_size=cfg.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_new = DataLoader(test_new, batch_size=cfg.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ------------------- Build models -------------------
    print("\nLoading models …")
    model_old = build_old(cfg.old_ckpt, device)
    model_new = build_new(cfg.new_ckpt, device)

    # ------------------- Evaluate OLD -------------------
    print("\n================  OLD MODEL  ================")
    y_val = np.asarray([label for _, label in val_old])  # labels are same for both loaders
    y_test = np.asarray([label for _, label in test_old])

    pred_val_old, conf_val_old = _get_preds(model_old, val_loader_old, device)
    pred_test_old, conf_test_old = _get_preds(model_old, test_loader_old, device)
    _print_metrics("Validation (old)", _evaluate_preds(y_val, pred_val_old))
    _print_metrics("Test (old)", _evaluate_preds(y_test, pred_test_old))

    # ------------------- Evaluate NEW -------------------
    print("\n================  NEW MODEL  ================")
    pred_val_new, conf_val_new = _get_preds(model_new, val_loader_new, device)
    pred_test_new, conf_test_new = _get_preds(model_new, test_loader_new, device)
    _print_metrics("Validation (new)", _evaluate_preds(y_val, pred_val_new))
    _print_metrics("Test (new)", _evaluate_preds(y_test, pred_test_new))

    # ------------------- Evaluate ENSEMBLE -------------------
    print("\n================ ENSEMBLE ===================")
    def _combine(p_old, c_old, p_new, c_new):
        agree = p_old == p_new
        combined = np.where(agree, p_old, p_old)  # disagreement → old
        return combined

    pred_val_ens = _combine(pred_val_old, conf_val_old, pred_val_new, conf_val_new)
    pred_test_ens = _combine(pred_test_old, conf_test_old, pred_test_new, conf_test_new)
    _print_metrics("Validation (ensemble)", _evaluate_preds(y_val, pred_val_ens))
    _print_metrics("Test (ensemble)", _evaluate_preds(y_test, pred_test_ens))


if __name__ == "__main__":
    main(sys.argv[1:])
