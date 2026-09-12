"""Controlled fold-0 comparison of two mathematically equivalent focal losses."""

from __future__ import annotations

import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data_loader import ProcessedCardioDataset
from .ecg_experiments import classification_metrics, f1_midpoint_threshold
from .ecg_model import ECG_Encoder
from .preprocessing import CHANNEL_ORDER, CHANNEL_ORDER_STRING, ECG_SAMPLES
from .train import FocalLoss as SharedFocalLoss


class OriginalNotebookFocalLoss(nn.Module):
    """Exact focal-loss calculation used in notebook 2.1."""

    def __init__(self, alpha=0.88, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        probabilities = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probabilities, 1 - probabilities)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


LOSS_IMPLEMENTATIONS = {
    "original_notebook_loss": OriginalNotebookFocalLoss,
    "shared_src_loss": SharedFocalLoss,
}


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _load_development(project_dir):
    data_dir = project_dir / "data/processed/patient_4ch/development"
    metadata = pd.read_csv(
        data_dir / "processed_metadata.csv", dtype={"Patient_ID": str}
    )
    if len(metadata) != 500 or metadata.Patient_ID.duplicated().any():
        raise ValueError("Expected 500 unique development patients.")
    if set(metadata.Fold.unique()) != set(range(5)):
        raise ValueError("Expected development folds 0 through 4.")
    if not metadata.Channel_Order.eq(CHANNEL_ORDER_STRING).all():
        raise ValueError(f"Expected site order {CHANNEL_ORDER_STRING}.")
    expected_labels = metadata.LVEF.le(0.40).astype(int)
    if not np.array_equal(metadata.Label.to_numpy(), expected_labels.to_numpy()):
        raise ValueError("Labels do not match the LVEF <= 40% rule.")
    train = metadata[metadata.Fold.ne(0)].reset_index(drop=True)
    validation = metadata[metadata.Fold.eq(0)].reset_index(drop=True)
    if len(train) != 400 or len(validation) != 100 or validation.Label.sum() != 12:
        raise ValueError("Expected 400 training and 100 fold-0 validation patients.")
    if set(train.Patient_ID) & set(validation.Patient_ID):
        raise ValueError("Training and validation patients overlap.")
    return train, validation, data_dir


def _make_loaders(train, validation, data_dir, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        ProcessedCardioDataset(train, data_dir, modality="ecg"),
        shuffle=True,
        generator=generator,
        **options,
    )
    validation_loader = DataLoader(
        ProcessedCardioDataset(validation, data_dir, modality="ecg"),
        shuffle=False,
        **options,
    )
    return train_loader, validation_loader, generator


@torch.no_grad()
def _predict(model, loader, device, criterion=None):
    model.eval()
    labels, probabilities = [], []
    total_loss = 0.0
    count = 0
    for values, batch_labels in loader:
        values = values.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).reshape(-1, 1)
        logits = model(values)
        if criterion is not None:
            total_loss += criterion(logits, batch_labels).item() * len(batch_labels)
        count += len(batch_labels)
        labels.extend(batch_labels.cpu().numpy().astype(int).ravel())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        total_loss / count if criterion is not None else float("nan"),
    )


def _train_one(
    train,
    validation,
    data_dir,
    loss_name,
    output_dir,
    *,
    seed=42,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-4,
    max_epochs=50,
    patience=10,
):
    _set_seed(seed)
    train_loader, validation_loader, shuffle_generator = _make_loaders(
        train, validation, data_dir, seed, batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {"in_channels": 4, "feature_dim": 128, "dropout": 0.3}
    model = ECG_Encoder(**model_config).to(device)
    initial_model_digest = _state_digest(model)
    initial_cpu_rng_digest = _tensor_digest(torch.get_rng_state())
    initial_gpu_rng_digests = (
        [_tensor_digest(value) for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else []
    )
    initial_shuffle_digest = _tensor_digest(shuffle_generator.get_state())

    sample, _ = next(iter(validation_loader))
    if tuple(sample.shape[1:]) != (4, ECG_SAMPLES):
        raise ValueError("Unexpected ECG input shape.")

    alpha = float(train.Label.eq(0).mean())
    criterion = LOSS_IMPLEMENTATIONS[loss_name](alpha=alpha, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_model.pt"
    history = []
    best_auroc = -float("inf")
    best_state = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for values, labels in train_loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is invalid.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)

        labels, probabilities, validation_loss = _predict(
            model, validation_loader, device, criterion
        )
        validation_auroc = roc_auc_score(labels, probabilities)
        validation_auprc = average_precision_score(labels, probabilities)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "val_loss": validation_loss,
            "val_auroc": validation_auroc,
            "val_auprc": validation_auprc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"{loss_name} | epoch {epoch} | loss {train_loss/train_count:.4f} | "
            f"validation AUROC {validation_auroc:.4f} | AUPRC {validation_auprc:.4f}"
        )
        scheduler.step(validation_auroc)
        if validation_auroc > best_auroc:
            best_auroc = validation_auroc
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    history = pd.DataFrame(history)
    history.to_csv(output_dir / "history.csv", index=False)
    if best_state is None:
        raise RuntimeError("Training did not produce a saved model.")
    model.load_state_dict(best_state)
    labels, probabilities, _ = _predict(model, validation_loader, device)
    if not np.array_equal(labels, validation.Label.to_numpy(dtype=int)):
        raise ValueError("Validation prediction order changed.")
    predictions = validation[["Patient_ID", "LVEF", "Label", "Fold"]].copy()
    predictions["loss_implementation"] = loss_name
    predictions["Probability"] = probabilities
    cutoff, _ = f1_midpoint_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, cutoff)
    predictions["Prediction"] = probabilities >= cutoff
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    best_epoch = int(history.loc[history.val_auroc.idxmax(), "epoch"])
    settings = {
        "loss_implementation": loss_name,
        "seed": seed,
        "validation_fold": 0,
        "training_folds": [1, 2, 3, 4],
        "model_config": model_config,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "focal_alpha": alpha,
        "focal_gamma": 2.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": elapsed,
        "initial_model_digest": initial_model_digest,
        "initial_cpu_rng_digest": initial_cpu_rng_digest,
        "initial_gpu_rng_digests": initial_gpu_rng_digests,
        "initial_shuffle_generator_digest": initial_shuffle_digest,
        "training_patient_ids": train.Patient_ID.tolist(),
        "validation_patient_ids": validation.Patient_ID.tolist(),
        "community_test_loaded": False,
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "model": loss_name,
        "result_type": "controlled_new_run",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": elapsed,
        **metrics,
    }, settings


def run_fold0_loss_comparison(project_dir):
    """Train the two controlled runs and save development-only comparisons."""
    project_dir = Path(project_dir).resolve()
    model_root = project_dir / "models/ecg_loss_comparison/seed42"
    result_root = project_dir / "analysis_outputs/ecg_loss_comparison/seed42"
    for path in (model_root, result_root):
        if path.exists():
            raise FileExistsError(
                f"Existing comparison output found: {path}. Nothing was overwritten."
            )
    model_root.mkdir(parents=True)
    result_root.mkdir(parents=True)
    train, validation, data_dir = _load_development(project_dir)

    new_rows = []
    setup_rows = []
    for loss_name in LOSS_IMPLEMENTATIONS:
        row, settings = _train_one(
            train, validation, data_dir, loss_name, model_root / loss_name
        )
        new_rows.append(row)
        setup_rows.append(settings)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    setup_equal = {
        "initial_model_state": setup_rows[0]["initial_model_digest"]
        == setup_rows[1]["initial_model_digest"],
        "initial_cpu_random_state": setup_rows[0]["initial_cpu_rng_digest"]
        == setup_rows[1]["initial_cpu_rng_digest"],
        "initial_gpu_random_state": setup_rows[0]["initial_gpu_rng_digests"]
        == setup_rows[1]["initial_gpu_rng_digests"],
        "initial_shuffle_state": setup_rows[0]["initial_shuffle_generator_digest"]
        == setup_rows[1]["initial_shuffle_generator_digest"],
        "training_patient_order": setup_rows[0]["training_patient_ids"]
        == setup_rows[1]["training_patient_ids"],
        "validation_patient_order": setup_rows[0]["validation_patient_ids"]
        == setup_rows[1]["validation_patient_ids"],
    }
    if not all(setup_equal.values()):
        raise AssertionError("The controlled runs did not start from the same setup.")

    historical = pd.read_csv(
        project_dir
        / "analysis_outputs/five_fold_cross_validation/ecg_fold0_audit"
        / "validation_replay_summary.csv"
    )
    historical = historical.assign(result_type="saved_checkpoint_replay")
    for column in ("threshold", "recall", "specificity", "balanced_accuracy",
                   "precision", "f1", "brier", "log_loss", "calibration_error",
                   "tn", "fp", "fn", "tp", "training_seconds"):
        if column not in historical:
            historical[column] = np.nan
    summary = pd.concat(
        [historical, pd.DataFrame(new_rows)], ignore_index=True, sort=False
    )
    summary.to_csv(result_root / "comparison_summary.csv", index=False)
    (result_root / "setup_checks.json").write_text(
        json.dumps({
            "all_equal": True,
            "checks": setup_equal,
            "development_only": True,
            "community_test_loaded": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_root
