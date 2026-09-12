"""Small, reproducible ECG experiments with one prediction per patient."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data_loader import ProcessedCardioDataset, get_dataloaders
from .ecg_model import ECG_Encoder, SharedSiteECGEncoder
from .preprocessing import CHANNEL_ORDER, CHANNEL_ORDER_STRING, ECG_SAMPLES
from .train import FocalLoss


MODEL_CONFIGS = {
    "joint_4ch": {"in_channels": 4, "feature_dim": 128, "dropout": 0.3},
    "shared_site": {"num_sites": 4, "feature_dim": 128, "dropout": 0.3},
}


class TrainingCropDataset(ProcessedCardioDataset):
    """Optionally shorten training ECGs while keeping all four sites together."""

    def __init__(self, metadata_df, data_dir, crop_mode="full", seed=42):
        super().__init__(metadata_df, data_dir, modality="ecg")
        if crop_mode not in ("full", "fixed_15s", "random_15s"):
            raise ValueError(f"Unknown crop mode: {crop_mode}")
        self.crop_mode = crop_mode
        self.crop_samples = ECG_SAMPLES // 2
        self.generator = torch.Generator().manual_seed(seed + 13579)

    def __getitem__(self, idx):
        ecg, label = super().__getitem__(idx)
        if self.crop_mode == "full":
            return ecg, label
        if self.crop_mode == "fixed_15s":
            start = (ECG_SAMPLES - self.crop_samples) // 2
        else:
            start = int(torch.randint(
                0, ECG_SAMPLES - self.crop_samples + 1, (1,),
                generator=self.generator,
            ))
        return ecg[:, start:start + self.crop_samples], label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(architecture):
    if architecture == "joint_4ch":
        return ECG_Encoder(**MODEL_CONFIGS[architecture])
    if architecture == "shared_site":
        return SharedSiteECGEncoder(**MODEL_CONFIGS[architecture])
    raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def validate_model_output(model, sample, device):
    """Check output shape without changing BatchNorm, dropout, or model mode."""
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(sample[:2].to(device))
    finally:
        model.train(previous_mode)
    if output.shape != (2, 1) or not torch.isfinite(output).all():
        raise ValueError("Model shape or finite-output check failed.")


def _labels_from_loader(loader):
    return loader.dataset.metadata["Label"].to_numpy(dtype=int)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    for ecg, batch_labels in loader:
        logits = model(ecg.to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced a nonfinite score.")
        probabilities.extend(logits.sigmoid().cpu().numpy().ravel())
        labels.extend(batch_labels.numpy().astype(int).ravel())
    return np.asarray(labels), np.asarray(probabilities)


def f1_midpoint_threshold(labels, probabilities):
    """Choose the best F1 classification, then place the cutoff between scores."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both classes.")
    unique_scores = np.unique(probabilities)
    candidates = np.r_[np.nextafter(unique_scores.min(), -np.inf), unique_scores]
    f1_values = np.asarray([
        f1_score(labels, probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ])
    best_index = int(np.flatnonzero(f1_values == f1_values.max())[-1])
    selected = float(candidates[best_index])
    predicted = probabilities >= selected
    positive_scores = probabilities[predicted]
    negative_scores = probabilities[~predicted]
    if len(positive_scores) and len(negative_scores):
        threshold = float((positive_scores.min() + negative_scores.max()) / 2)
    elif len(positive_scores):
        threshold = float(np.nextafter(positive_scores.min(), -np.inf))
    else:
        threshold = float(np.nextafter(probabilities.max(), np.inf))
    if not np.array_equal(predicted, probabilities >= threshold):
        raise AssertionError("Midpoint changed the selected classifications.")
    return threshold, float(f1_values[best_index])


def classification_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.minimum(np.digitize(probabilities, bin_edges[1:-1]), 9)
    calibration_error = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            calibration_error += mask.mean() * abs(
                probabilities[mask].mean() - labels[mask].mean()
            )
    return {
        "patients": int(len(labels)),
        "hfref_patients": int(labels.sum()),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "calibration_error": float(calibration_error),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _validate_fold_roles(metadata, train_folds, tuning_fold, scoring_fold):
    train_df = metadata[metadata["Fold"].isin(train_folds)].reset_index(drop=True)
    tune_df = metadata[metadata["Fold"].eq(tuning_fold)].reset_index(drop=True)
    score_df = metadata[metadata["Fold"].eq(scoring_fold)].reset_index(drop=True)
    role_ids = [set(frame.Patient_ID.astype(str)) for frame in (train_df, tune_df, score_df)]
    if any(role_ids[left] & role_ids[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Training, tuning, and scoring patients must be separate.")
    if set(train_folds) | {int(tuning_fold), int(scoring_fold)} != set(metadata.Fold.unique()):
        raise ValueError("Every development fold must have exactly one experiment role.")
    for frame in (train_df, tune_df, score_df):
        if frame.empty or set(frame.Label.unique()) != {0, 1}:
            raise ValueError("Every fold role must contain both classes.")
        if not frame.Channel_Order.eq(CHANNEL_ORDER_STRING).all():
            raise ValueError("Unexpected site order.")
    return train_df, tune_df, score_df


def _make_ecg_loader(
    metadata, data_dir, batch_size, *, shuffle, seed, crop_mode="full"
):
    dataset = TrainingCropDataset(metadata, data_dir, crop_mode=crop_mode, seed=seed)
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    if shuffle:
        options["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, **options)


def train_tune_score_experiment(
    metadata,
    data_dir,
    architecture,
    train_folds,
    tuning_fold,
    scoring_fold,
    output_dir,
    *,
    seed=42,
    batch_size=16,
    max_epochs=50,
    patience=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
    training_crop="full",
    device=None,
):
    """Train, select on a tuning fold, then score a separate fold once."""
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, tune_df, score_df = _validate_fold_roles(
        metadata, train_folds, tuning_fold, scoring_fold
    )
    train_loader = _make_ecg_loader(
        train_df, data_dir, batch_size, shuffle=True, seed=seed,
        crop_mode=training_crop,
    )
    tune_loader = _make_ecg_loader(
        tune_df, data_dir, batch_size, shuffle=False, seed=seed
    )
    score_loader = _make_ecg_loader(
        score_df, data_dir, batch_size, shuffle=False, seed=seed
    )

    model = make_model(architecture).to(device)
    sample, _ = next(iter(tune_loader))
    if tuple(sample.shape[1:]) != (4, ECG_SAMPLES):
        raise ValueError(f"Unexpected ECG shape: {tuple(sample.shape)}")
    validate_model_output(model, sample, device)

    alpha = float(train_df.Label.eq(0).mean())
    criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    best_auroc = -float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        labels, probabilities = predict(model, tune_loader, device)
        tune_metrics = classification_metrics(labels, probabilities, 0.5)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "tune_auroc": tune_metrics["auroc"],
            "tune_auprc": tune_metrics["auprc"],
            "tune_brier": tune_metrics["brier"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
        print(
            f"score fold {scoring_fold}, {architecture}, epoch {epoch}: "
            f"tuning AUPRC {tune_metrics['auprc']:.4f}, "
            f"AUROC {tune_metrics['auroc']:.4f}"
        )
        scheduler.step(tune_metrics["auroc"])
        if tune_metrics["auroc"] > best_auroc:
            best_auroc = tune_metrics["auroc"]
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    training_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    reloaded = make_model(architecture).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    tune_labels, tune_probabilities = predict(reloaded, tune_loader, device)
    threshold, selected_f1 = f1_midpoint_threshold(tune_labels, tune_probabilities)
    score_labels, score_probabilities = predict(reloaded, score_loader, device)
    tune_metrics = classification_metrics(tune_labels, tune_probabilities, threshold)
    score_metrics = classification_metrics(score_labels, score_probabilities, threshold)

    history_df = pd.DataFrame(history)
    best_epoch = int(history_df.loc[history_df.tune_auroc.idxmax(), "epoch"])
    settings = {
        **score_metrics,
        "architecture": architecture,
        "seed": seed,
        "train_folds": list(train_folds),
        "tuning_fold": int(tuning_fold),
        "scoring_fold": int(scoring_fold),
        "train_patients": int(len(train_df)),
        "tuning_patients": int(len(tune_df)),
        "scoring_patients": int(len(score_df)),
        "best_epoch": best_epoch,
        "epochs_run": int(len(history)),
        "best_tuning_auroc": best_auroc,
        "tuning_auprc": tune_metrics["auprc"],
        "tuning_auroc": tune_metrics["auroc"],
        "tuning_f1": selected_f1,
        "parameter_count": count_parameters(model),
        "training_seconds": training_seconds,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "device": str(device),
        "batch_size": batch_size,
        "focal_alpha": alpha,
        "focal_gamma": 2.0,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "site_order": list(CHANNEL_ORDER),
        "training_crop": training_crop,
        "tuning_and_scoring_input": "full_30s",
        "shape_check_mode": "evaluation_without_gradients",
    }
    for role, frame, probabilities in (
        ("tuning", tune_df, tune_probabilities),
        ("scoring", score_df, score_probabilities),
    ):
        predictions = frame[["Patient_ID", "Fold", "Label", "LVEF"]].copy()
        predictions["Probability"] = probabilities
        predictions["Prediction"] = probabilities >= threshold
        predictions["Threshold"] = threshold
        predictions["Architecture"] = architecture
        predictions.to_csv(output_dir / f"{role}_predictions.csv", index=False)
    history_df.to_csv(output_dir / "history.csv", index=False)
    (output_dir / "settings_and_results.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False), encoding="utf-8"
    )
    return settings


def paired_bootstrap_differences(predictions, n_bootstrap=2000, seed=24680):
    """Patient-paired uncertainty for shared-site minus joint-model metrics."""
    architectures = ["joint_4ch", "shared_site"]
    indexed = {
        architecture: predictions[predictions.Architecture.eq(architecture)]
        .sort_values("Patient_ID").set_index("Patient_ID")
        for architecture in architectures
    }
    if not indexed[architectures[0]].index.equals(indexed[architectures[1]].index):
        raise ValueError("Architectures must score the same patients.")
    base = indexed[architectures[0]]
    rng = np.random.default_rng(seed)
    metric_names = [
        "auprc", "auroc", "recall", "specificity",
        "balanced_accuracy", "brier", "calibration_error",
    ]
    folds = sorted(base.Fold.unique())
    observed = {}
    for architecture in architectures:
        observed_fold_metrics = []
        for fold in folds:
            frame = indexed[architecture][indexed[architecture].Fold.eq(fold)]
            observed_fold_metrics.append(classification_metrics(
                frame.Label.to_numpy(), frame.Probability.to_numpy(),
                float(frame.Threshold.iloc[0]),
            ))
        observed[architecture] = {
            name: float(np.mean([values[name] for values in observed_fold_metrics]))
            for name in metric_names
        }
    bootstrap_metrics = {
        architecture: {
            name: np.zeros(n_bootstrap, dtype=float) for name in metric_names
        }
        for architecture in architectures
    }
    for fold in folds:
        fold_base = base[base.Fold.eq(fold)]
        fold_ids = fold_base.index.to_numpy()
        fold_labels = fold_base.Label.to_numpy(dtype=int)
        negative_positions = np.flatnonzero(fold_labels == 0)
        positive_positions = np.flatnonzero(fold_labels == 1)
        draws = np.concatenate([
            rng.choice(
                negative_positions,
                size=(n_bootstrap, len(negative_positions)),
                replace=True,
            ),
            rng.choice(
                positive_positions,
                size=(n_bootstrap, len(positive_positions)),
                replace=True,
            ),
        ], axis=1)
        sampled_labels = fold_labels[draws]
        negative_count = len(negative_positions)
        positive_count = len(positive_positions)
        for architecture in architectures:
            frame = indexed[architecture].loc[fold_ids]
            sampled_probabilities = frame.Probability.to_numpy(dtype=float)[draws]
            sampled_predictions = frame.Prediction.to_numpy(dtype=bool)[draws]

            recall = sampled_predictions[:, negative_count:].mean(axis=1)
            specificity = (~sampled_predictions[:, :negative_count]).mean(axis=1)
            brier = np.mean((sampled_probabilities - sampled_labels) ** 2, axis=1)

            positive_probabilities = sampled_probabilities[:, negative_count:]
            negative_probabilities = sampled_probabilities[:, :negative_count]
            pairwise = positive_probabilities[:, :, None] - negative_probabilities[:, None, :]
            auroc = (pairwise > 0).mean(axis=(1, 2))
            auroc += 0.5 * (pairwise == 0).mean(axis=(1, 2))

            order = np.argsort(-sampled_probabilities, axis=1, kind="stable")
            sorted_labels = np.take_along_axis(sampled_labels, order, axis=1)
            cumulative_positives = np.cumsum(sorted_labels, axis=1)
            ranks = np.arange(1, sampled_labels.shape[1] + 1)
            auprc = np.sum(
                (cumulative_positives / ranks) * sorted_labels, axis=1
            ) / positive_count

            calibration_error = np.zeros(n_bootstrap, dtype=float)
            bin_ids = np.minimum(
                np.digitize(sampled_probabilities, np.linspace(0, 1, 11)[1:-1]), 9
            )
            for bin_id in range(10):
                mask = bin_ids == bin_id
                counts = mask.sum(axis=1)
                valid = counts > 0
                probability_means = np.zeros(n_bootstrap, dtype=float)
                label_means = np.zeros(n_bootstrap, dtype=float)
                probability_means[valid] = (
                    (sampled_probabilities * mask).sum(axis=1)[valid] / counts[valid]
                )
                label_means[valid] = (
                    (sampled_labels * mask).sum(axis=1)[valid] / counts[valid]
                )
                calibration_error += (
                    counts / sampled_labels.shape[1]
                ) * np.abs(probability_means - label_means)

            values = {
                "auprc": auprc,
                "auroc": auroc,
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": (recall + specificity) / 2,
                "brier": brier,
                "calibration_error": calibration_error,
            }
            for name in metric_names:
                bootstrap_metrics[architecture][name] += values[name] / len(folds)
    samples = {
        name: bootstrap_metrics["shared_site"][name]
        - bootstrap_metrics["joint_4ch"][name]
        for name in metric_names
    }
    return pd.DataFrame([
        {
            "metric": name,
            "difference_shared_minus_joint": (
                observed["shared_site"][name] - observed["joint_4ch"][name]
            ),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_samples": n_bootstrap,
        }
        for name, values in samples.items()
    ])


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for ecg, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ecg)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became nonfinite.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count


def train_experiment(
    metadata,
    data_dir,
    architecture,
    train_folds,
    validation_fold,
    output_dir,
    *,
    seed=42,
    batch_size=16,
    max_epochs=50,
    patience=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device=None,
):
    """Train one fixed comparison and save its complete development record."""
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = metadata[metadata["Fold"].isin(train_folds)].reset_index(drop=True)
    val_df = metadata[metadata["Fold"].eq(validation_fold)].reset_index(drop=True)
    if set(train_df.Patient_ID) & set(val_df.Patient_ID):
        raise ValueError("Training and validation patients overlap.")
    if not train_df.Channel_Order.eq(CHANNEL_ORDER_STRING).all():
        raise ValueError("Unexpected site order.")
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, data_dir, batch_size=batch_size, modality="ecg", seed=seed
    )
    model = make_model(architecture).to(device)
    sample, _ = next(iter(val_loader))
    if tuple(sample.shape[1:]) != (4, ECG_SAMPLES):
        raise ValueError(f"Unexpected ECG shape: {tuple(sample.shape)}")
    validate_model_output(model, sample, device)

    alpha = float(train_df.Label.eq(0).mean())
    criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    best_auroc = -float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        labels, probabilities = predict(model, val_loader, device)
        validation_loss = float(np.mean(
            -alpha * labels * (1 - probabilities) ** 2 * np.log(np.clip(probabilities, 1e-8, 1))
            -(1 - alpha) * (1 - labels) * probabilities ** 2 * np.log(np.clip(1 - probabilities, 1e-8, 1))
        ))
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "val_auroc": auroc,
            "val_auprc": auprc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
        print(
            f"{architecture} epoch {epoch}: loss {train_loss:.4f}, "
            f"validation AUPRC {auprc:.4f}, AUROC {auroc:.4f}"
        )
        scheduler.step(auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    total_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    reloaded = make_model(architecture).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    labels, probabilities = predict(reloaded, val_loader, device)
    threshold, selected_f1 = f1_midpoint_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, threshold)
    metrics.update({
        "architecture": architecture,
        "seed": seed,
        "train_folds": list(train_folds),
        "validation_fold": int(validation_fold),
        "train_patients": int(len(train_df)),
        "validation_patients": int(len(val_df)),
        "best_epoch": int(pd.DataFrame(history).val_auroc.idxmax() + 1),
        "epochs_run": int(len(history)),
        "best_validation_auroc": best_auroc,
        "selected_f1_check": selected_f1,
        "parameter_count": count_parameters(model),
        "training_seconds": total_seconds,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "device": str(device),
        "batch_size": batch_size,
        "focal_alpha": alpha,
        "focal_gamma": 2.0,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "site_order": list(CHANNEL_ORDER),
        "shape_check_mode": "evaluation_without_gradients",
    })
    predictions = val_df[["Patient_ID", "Fold", "Label", "LVEF"]].copy()
    predictions["Probability"] = probabilities
    predictions["Prediction"] = probabilities >= threshold
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "settings_and_results.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def replay_historical(project_dir, output_dir, device=None):
    project_dir = Path(project_dir)
    settings = json.loads(
        (project_dir / "models/best_ecg_4ch_fold0.json").read_text(encoding="utf-8")
    )
    metadata = pd.read_csv(
        project_dir / "data/processed/patient_4ch/development/processed_metadata.csv",
        dtype={"Patient_ID": str},
    )
    validation = metadata.set_index("Patient_ID").loc[
        settings["validation_patient_ids"]
    ].reset_index()
    dataset = ProcessedCardioDataset(
        validation,
        project_dir / "data/processed/patient_4ch/development",
        modality="ecg",
    )
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ECG_Encoder(**settings["model_config"]).to(device)
    model.load_state_dict(torch.load(
        project_dir / "models/best_ecg_4ch_fold0.pt",
        map_location=device,
        weights_only=True,
    ))
    labels, probabilities = predict(model, loader, device)
    metrics = classification_metrics(labels, probabilities, settings["threshold"])
    metrics.update({"architecture": "historical_joint_4ch", "device": str(device)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = validation[["Patient_ID", "Fold", "Label", "LVEF"]].copy()
    result["Probability"] = probabilities
    result["Prediction"] = probabilities >= settings["threshold"]
    result.to_csv(output_dir / "validation_predictions.csv", index=False)
    (output_dir / "settings_and_results.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def run_fold0(project_dir):
    project_dir = Path(project_dir)
    metadata = pd.read_csv(
        project_dir / "data/processed/patient_4ch/development/processed_metadata.csv",
        dtype={"Patient_ID": str},
    )
    artifact_root = project_dir / "models/ecg_experiments/fold0_seed42"
    result_root = project_dir / "analysis_outputs/ecg_experiments"
    result_root.mkdir(parents=True, exist_ok=True)
    replay_dir = result_root / "historical_fold0_replay"
    replay_result = replay_dir / "settings_and_results.json"
    if replay_result.is_file():
        results = [json.loads(replay_result.read_text(encoding="utf-8"))]
    else:
        results = [replay_historical(project_dir, replay_dir)]
    for architecture in ("joint_4ch", "shared_site"):
        results.append(train_experiment(
            metadata,
            project_dir / "data/processed/patient_4ch/development",
            architecture,
            train_folds=(1, 2, 3, 4),
            validation_fold=0,
            output_dir=artifact_root / architecture,
        ))
    table = pd.DataFrame(results)
    table.to_csv(result_root / "fold0_architecture_comparison.csv", index=False)
    print(table[[
        "architecture", "auprc", "auroc", "recall", "specificity",
        "balanced_accuracy", "f1", "brier", "parameter_count", "training_seconds"
    ]].to_string(index=False))


def run_five_fold(project_dir, seed=42, n_bootstrap=2000, run_label=None):
    """Give each patient one score from a fold never used for model selection."""
    project_dir = Path(project_dir)
    metadata = pd.read_csv(
        project_dir / "data/processed/patient_4ch/development/processed_metadata.csv",
        dtype={"Patient_ID": str},
    )
    if metadata.Patient_ID.duplicated().any() or len(metadata) != 500:
        raise ValueError("Expected 500 unique development patients.")
    folds = sorted(int(value) for value in metadata.Fold.unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected folds 0 to 4; found {folds}.")
    run_name = "five_fold"
    if run_label:
        if not str(run_label).replace("_", "").isalnum():
            raise ValueError("run_label may contain only letters, numbers, and underscores.")
        run_name += f"_{run_label}"
    run_name += f"_seed{seed}"
    artifact_root = project_dir / "models/ecg_experiments" / run_name
    result_root = project_dir / "analysis_outputs/ecg_experiments"
    result_root.mkdir(parents=True, exist_ok=True)
    results = []
    prediction_frames = []
    for scoring_fold in folds:
        tuning_fold = (scoring_fold + 1) % len(folds)
        train_folds = [
            fold for fold in folds if fold not in (scoring_fold, tuning_fold)
        ]
        for architecture in ("joint_4ch", "shared_site"):
            output_dir = artifact_root / f"score_fold_{scoring_fold}" / architecture
            settings_path = output_dir / "settings_and_results.json"
            if settings_path.is_file():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            else:
                settings = train_tune_score_experiment(
                    metadata,
                    project_dir / "data/processed/patient_4ch/development",
                    architecture,
                    train_folds=train_folds,
                    tuning_fold=tuning_fold,
                    scoring_fold=scoring_fold,
                    output_dir=output_dir,
                    seed=seed,
                )
            results.append(settings)
            prediction_frames.append(pd.read_csv(
                output_dir / "scoring_predictions.csv", dtype={"Patient_ID": str}
            ))
    results_df = pd.DataFrame(results).sort_values(
        ["scoring_fold", "architecture"]
    ).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    counts = predictions.groupby("Architecture").Patient_ID.agg(["count", "nunique"])
    if not counts.eq(500).all().all():
        raise ValueError(f"Each architecture must score 500 unique patients:\n{counts}")
    patient_sets = predictions.groupby("Architecture").Patient_ID.apply(set)
    if patient_sets.iloc[0] != patient_sets.iloc[1]:
        raise ValueError("Architectures did not score identical patient sets.")

    results_path = result_root / f"{run_name}_results.csv"
    predictions_path = result_root / f"{run_name}_predictions.csv"
    uncertainty_path = result_root / f"{run_name}_paired_uncertainty.csv"
    summary_path = result_root / f"{run_name}_summary.csv"
    results_df.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metric_columns = [
        "auprc", "auroc", "recall", "specificity", "balanced_accuracy",
        "precision", "f1", "brier", "log_loss", "calibration_error",
        "training_seconds", "best_epoch",
    ]
    summary = results_df.groupby("architecture")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(summary_path, index=False)
    uncertainty = paired_bootstrap_differences(
        predictions, n_bootstrap=n_bootstrap, seed=seed + 24680
    )
    uncertainty.to_csv(uncertainty_path, index=False)
    print("\nMean results across five separate scoring folds:")
    print(results_df.groupby("architecture")[[
        "auprc", "auroc", "recall", "specificity", "balanced_accuracy",
        "f1", "brier", "calibration_error",
    ]].mean().to_string())
    print("\nPaired patient bootstrap, shared-site minus joint model:")
    print(uncertainty.to_string(index=False))
    return results_df, predictions, uncertainty


def run_crop_comparison(project_dir, seed=42, n_bootstrap=2000):
    """Compare full, fixed 15-second, and random 15-second training inputs."""
    project_dir = Path(project_dir)
    metadata = pd.read_csv(
        project_dir / "data/processed/patient_4ch/development/processed_metadata.csv",
        dtype={"Patient_ID": str},
    )
    folds = [0, 1, 2, 3, 4]
    result_root = project_dir / "analysis_outputs/ecg_experiments"
    artifact_root = project_dir / f"models/ecg_experiments/crop_comparison_seed{seed}"
    results = []
    predictions = []
    for scoring_fold in folds:
        tuning_fold = (scoring_fold + 1) % 5
        train_folds = [fold for fold in folds if fold not in (scoring_fold, tuning_fold)]
        for crop_mode in ("fixed_15s", "random_15s"):
            output_dir = artifact_root / f"score_fold_{scoring_fold}" / crop_mode
            settings_path = output_dir / "settings_and_results.json"
            if settings_path.is_file():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            else:
                settings = train_tune_score_experiment(
                    metadata,
                    project_dir / "data/processed/patient_4ch/development",
                    "joint_4ch",
                    train_folds=train_folds,
                    tuning_fold=tuning_fold,
                    scoring_fold=scoring_fold,
                    output_dir=output_dir,
                    seed=seed,
                    training_crop=crop_mode,
                )
            results.append(settings)
            frame = pd.read_csv(
                output_dir / "scoring_predictions.csv", dtype={"Patient_ID": str}
            )
            frame["Training_Crop"] = crop_mode
            predictions.append(frame)

        full_dir = (
            project_dir / f"models/ecg_experiments/five_fold_seed{seed}"
            / f"score_fold_{scoring_fold}" / "joint_4ch"
        )
        full_settings = json.loads(
            (full_dir / "settings_and_results.json").read_text(encoding="utf-8")
        )
        full_settings["training_crop"] = "full"
        results.append(full_settings)
        full_frame = pd.read_csv(
            full_dir / "scoring_predictions.csv", dtype={"Patient_ID": str}
        )
        full_frame["Training_Crop"] = "full"
        predictions.append(full_frame)

    results_df = pd.DataFrame(results).sort_values(
        ["scoring_fold", "training_crop"]
    ).reset_index(drop=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    counts = predictions_df.groupby("Training_Crop").Patient_ID.agg(["count", "nunique"])
    if not counts.eq(500).all().all():
        raise ValueError(f"Every crop condition must score 500 unique patients:\n{counts}")
    result_root.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        result_root / f"crop_comparison_seed{seed}_results.csv", index=False
    )
    predictions_df.to_csv(
        result_root / f"crop_comparison_seed{seed}_predictions.csv", index=False
    )
    summary = results_df.groupby("training_crop")[[
        "auprc", "auroc", "recall", "specificity", "balanced_accuracy",
        "f1", "brier", "calibration_error", "training_seconds",
    ]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        result_root / f"crop_comparison_seed{seed}_summary.csv", index=False
    )
    print("\nMean results across five scoring folds:")
    print(results_df.groupby("training_crop")[[
        "auprc", "auroc", "recall", "specificity", "balanced_accuracy",
        "f1", "brier", "calibration_error",
    ]].mean().to_string())
    return results_df, predictions_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["replay", "fold0", "five-fold", "five-fold-corrected", "crop-comparison"],
    )
    parser.add_argument("--project-dir", default=str(Path.cwd()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()
    if args.command == "replay":
        metrics = replay_historical(
            project_dir,
            project_dir / "analysis_outputs/ecg_experiments/historical_fold0_replay",
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "fold0":
        run_fold0(project_dir)
    elif args.command == "five-fold":
        run_five_fold(project_dir, seed=args.seed, n_bootstrap=args.bootstrap)
    elif args.command == "five-fold-corrected":
        run_five_fold(
            project_dir, seed=args.seed, n_bootstrap=args.bootstrap,
            run_label="corrected",
        )
    else:
        run_crop_comparison(project_dir, seed=args.seed, n_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
