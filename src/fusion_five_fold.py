"""Frozen-encoder concatenation fusion, evaluated with the same five folds
used for the ECG-only and PCG-only baselines in five_fold_validation.py.

Each fusion fold reuses *that fold's own* pretrained ECG and PCG encoders
(trained on the other four folds, saved by
`five_fold_validation.train_five_fold_models`) and only trains the small
fusion classifier head on top by default (`freeze_encoders=True`). This keeps
the comparison to the single-modality OOF results apples-to-apples: same
folds, same patients, same held-out validation set per fold.

Run `train_five_fold_models` for both "ecg" and "pcg" with `encoder_seed`
before calling `train_fusion_five_fold`.
"""

from __future__ import annotations

import json
from platform import architecture
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data_loader import ProcessedCardioDataset
from .ecg_cutoff_analysis import specificity_target_threshold
from .ecg_experiments import classification_metrics, f1_midpoint_threshold
from .five_fold_validation import FOLDS, _load_metadata, _set_seed
from .fusion_model import build_fusion_model
from .train import FocalLoss


def _fusion_paths(
    project_dir: Path,
    seed: int,
    architecture: str,
) -> tuple[Path, Path]:

    architecture = architecture.lower()

    if architecture not in ("concat", "attention"):
        raise ValueError(
            "architecture must be 'concat' or 'attention'."
        )

    model_dir = (
        project_dir
        / "models"
        / "five_fold_cross_validation"
        / f"fusion_{architecture}_seed{seed}"
    )

    result_dir = (
        project_dir
        / "analysis_outputs"
        / "five_fold_cross_validation"
        / f"fusion_{architecture}_seed{seed}"
    )

    return model_dir, result_dir

def _encoder_checkpoint(project_dir, modality, encoder_seed, fold):
    fold_dir = (
        project_dir / "models" / "five_fold_cross_validation"
        / f"{modality}_seed{encoder_seed}" / f"fold_{fold}"
    )
    checkpoint = fold_dir / "best_model.pt"
    settings_path = fold_dir / "settings.json"
    if not checkpoint.is_file() or not settings_path.is_file():
        raise FileNotFoundError(
            f"Missing {modality} fold-{fold} checkpoint under {fold_dir}. "
            "Run five_fold_validation.train_five_fold_models for this "
            "modality and encoder_seed first."
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    return checkpoint, settings["model_config"]


def _cutoffs(labels, probabilities):
    f1_cutoff, _ = f1_midpoint_threshold(labels, probabilities)
    specificity_cutoff, specificity_metrics = specificity_target_threshold(
        labels, probabilities, target=0.90
    )
    return {
        "maximum_development_f1": (f1_cutoff, None),
        "target_90pct_development_specificity": (
            specificity_cutoff, specificity_metrics["specificity"]
        ),
    }


@torch.no_grad()
def _predict_fusion(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for ecg, pcg, batch_labels in loader:
        logits = model(
            ecg.to(device, non_blocking=True), pcg.to(device, non_blocking=True)
        )
        if tuple(logits.shape) != (len(batch_labels), 1):
            raise ValueError("Expected one fusion output per patient.")
        if not torch.isfinite(logits).all():
            raise ValueError("Fusion model produced an invalid output.")
        labels.extend(batch_labels.numpy().astype(int))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def _make_fusion_loaders(train_df, validation_df, data_dir, seed, batch_size):
    train_dataset = ProcessedCardioDataset(train_df, data_dir, modality="both")
    validation_dataset = ProcessedCardioDataset(validation_df, data_dir, modality="both")
    if set(train_df.Patient_ID) & set(validation_df.Patient_ID):
        raise ValueError("Training and validation patients overlap.")
    options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **options),
        DataLoader(validation_dataset, shuffle=False, **options),
    )


def _train_one_fusion_fold(
    project_dir,
    encoder_seed,
    fusion_seed,
    validation_fold,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    freeze_encoders,
    architecture,
    hidden_dim,
    dropout,
    attention_dim,
    num_heads,
    model_root,
):
    metadata, data_dir = _load_metadata(project_dir, "development")
    train_df = metadata[metadata.Fold.ne(validation_fold)].reset_index(drop=True)
    validation_df = metadata[metadata.Fold.eq(validation_fold)].reset_index(drop=True)
    if len(train_df) != 400 or len(validation_df) != 100:
        raise ValueError("Each run requires 400 training and 100 validation patients.")
    if int(validation_df.Label.sum()) != 12:
        raise ValueError("Expected 12 HFrEF patients in every validation fold.")

    fold_seed = fusion_seed + validation_fold
    _set_seed(fold_seed)
    train_loader, validation_loader = _make_fusion_loaders(
        train_df, validation_df, data_dir, fold_seed, batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ecg_checkpoint, ecg_config = _encoder_checkpoint(
        project_dir, "ecg", encoder_seed, validation_fold
    )
    pcg_checkpoint, pcg_config = _encoder_checkpoint(
        project_dir, "pcg", encoder_seed, validation_fold
    )
    model = build_fusion_model(
        ecg_checkpoint,
        ecg_config,
        pcg_checkpoint,
        pcg_config,
        architecture=architecture,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoders=freeze_encoders,
        attention_dim=attention_dim,
        num_heads=num_heads,
        device=device,
    )

    sample_ecg, sample_pcg, _ = next(iter(validation_loader))
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        checked_output = model(sample_ecg[:1].to(device), sample_pcg[:1].to(device))
    model.train(previous_mode)
    if tuple(checked_output.shape) != (1, 1):
        raise ValueError("Fusion model shape check failed.")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Fusion model has no trainable parameters.")

    focal_alpha = float(train_df.Label.eq(0).mean())
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    checkpoint_dir = model_root / f"fold_{validation_fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    best_auroc = -float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for ecg, pcg, labels in train_loader:
            ecg = ecg.to(device, non_blocking=True)
            pcg = pcg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ecg, pcg)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is invalid.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)
        if train_count == 0:
            raise ValueError("Cannot train with an empty loader.")

        validation_labels, validation_probabilities = _predict_fusion(
            model, validation_loader, device
        )
        validation_auroc = roc_auc_score(validation_labels, validation_probabilities)
        validation_auprc = average_precision_score(
            validation_labels, validation_probabilities
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "val_auroc": validation_auroc,
            "val_auprc": validation_auprc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"FUSION fold {validation_fold} | epoch {epoch} | "
            f"train loss {train_loss / train_count:.4f} | "
            f"validation AUROC {validation_auroc:.4f} | "
            f"average precision {validation_auprc:.4f}"
        )
        scheduler.step(validation_auroc)
        if validation_auroc > best_auroc:
            best_auroc = validation_auroc
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    training_seconds = time.perf_counter() - started
    if best_state is None or not checkpoint_path.is_file():
        raise RuntimeError("Training did not save a fusion checkpoint.")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    validation_labels, validation_probabilities = _predict_fusion(
        model, validation_loader, device
    )
    if not np.array_equal(validation_labels, validation_df.Label.to_numpy()):
        raise ValueError("Validation prediction order changed.")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(checkpoint_dir / "history.csv", index=False)
    predictions = validation_df[["Patient_ID", "LVEF", "Label", "Fold"]].copy()
    predictions["Probability"] = validation_probabilities
    predictions.to_csv(checkpoint_dir / "validation_predictions.csv", index=False)

    best_epoch = int(history_frame.loc[history_frame.val_auroc.idxmax(), "epoch"])
    fold_settings = {
        "architecture": architecture,
        "attention_dim": attention_dim,
        "num_heads": num_heads,
        "freeze_encoders": freeze_encoders,
        "encoder_seed": encoder_seed,
        "encoder_source_fold": validation_fold,
        "ecg_checkpoint": str(ecg_checkpoint),
        "pcg_checkpoint": str(pcg_checkpoint),
        "ecg_model_config": ecg_config,
        "pcg_model_config": pcg_config,
        "fusion_seed": fusion_seed,
        "fold_seed": fold_seed,
        "validation_fold": validation_fold,
        "training_folds": [fold for fold in FOLDS if fold != validation_fold],
        "training_patients": len(train_df),
        "validation_patients": len(validation_df),
        "training_hfref_patients": int(train_df.Label.sum()),
        "validation_hfref_patients": int(validation_df.Label.sum()),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "focal_alpha": focal_alpha,
        "focal_gamma": 2.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_validation_auroc": float(best_auroc),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else float("nan")
        ),
    }
    (checkpoint_dir / "settings.json").write_text(
        json.dumps(fold_settings, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, history_frame, fold_settings


def train_fusion_five_fold(
    project_dir,
    architecture="concat",
    encoder_seed=42,
    fusion_seed=42,
    batch_size=16,
    learning_rate=1e-3,
    max_epochs=50,
    patience=10,
    freeze_encoders=True,
    hidden_dim=128,
    dropout=0.3,
    attention_dim=128,
    num_heads=4,
):
    """Train five fusion heads, one per fold, and save one OOF prediction per patient.

    Each fold reuses that fold's own pretrained ECG and PCG encoders from
    `five_fold_validation.train_five_fold_models`; run that for both
    modalities with `encoder_seed` first. With `freeze_encoders=True`
    (default) only the small classifier head is trained; the encoders keep
    the weights they already learned on their own four training folds.
    """
    project_dir = Path(project_dir).resolve()
    model_root, result_root = _fusion_paths(project_dir, fusion_seed, architecture,)
    allowed_parents = {
        (project_dir / "models" / "five_fold_cross_validation").resolve(),
        (project_dir / "analysis_outputs" / "five_fold_cross_validation").resolve(),
    }
    for path in (model_root, result_root):
        if path.resolve().parent not in allowed_parents:
            raise ValueError(f"Unexpected generated-output path: {path}")
        if path.exists():
            shutil.rmtree(path)
    model_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    metadata, _ = _load_metadata(project_dir, "development")
    prediction_frames = []
    fold_setting_rows = []
    for validation_fold in FOLDS:
        predictions, _, fold_settings = _train_one_fusion_fold(
            project_dir=project_dir,
            encoder_seed=encoder_seed,
            fusion_seed=fusion_seed,
            validation_fold=validation_fold,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            freeze_encoders=freeze_encoders,
            hidden_dim=hidden_dim,
            dropout=dropout,
            model_root=model_root,
            architecture=architecture,
            attention_dim=attention_dim,
            num_heads=num_heads,
        )
        prediction_frames.append(predictions)
        fold_setting_rows.append(fold_settings)

    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["Fold", "Patient_ID"]
    ).reset_index(drop=True)
    if len(oof) != 500 or oof.Patient_ID.duplicated().any():
        raise ValueError("Expected one validation prediction for every patient.")
    if set(oof.Patient_ID) != set(metadata.Patient_ID):
        raise ValueError("Out-of-fold patient IDs do not match development metadata.")
    if not np.isfinite(oof.Probability).all():
        raise ValueError("Out-of-fold probabilities are invalid.")

    selected_cutoffs = _cutoffs(oof.Label, oof.Probability)
    development_rows = []
    fold_rows = []
    for rule, (cutoff, achieved_specificity) in selected_cutoffs.items():
        metrics = classification_metrics(oof.Label, oof.Probability, cutoff)
        development_rows.append({
            "modality": "fusion",
        "architecture": architecture,
            "seed": fusion_seed,
            "cutoff_rule": rule,
            "development_cutoff": cutoff,
            "cutoff_selection_specificity": achieved_specificity,
            **metrics,
        })
        oof[f"Prediction_{rule}"] = oof.Probability.ge(cutoff)
        for fold, frame in oof.groupby("Fold"):
            fold_metrics = classification_metrics(frame.Label, frame.Probability, cutoff)
            fold_rows.append({
                "modality": "fusion",
                "architecture": architecture,
                "seed": fusion_seed,
                "validation_fold": int(fold),
                "cutoff_rule": rule,
                "development_cutoff": cutoff,
                **fold_metrics,
            })

    oof.to_csv(result_root / "development_oof_predictions.csv", index=False)
    pd.DataFrame(development_rows).to_csv(
        result_root / "development_summary.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(result_root / "fold_results.csv", index=False)
    run_settings = {
        "architecture": architecture,
        "attention_dim": attention_dim,
        "num_heads": num_heads,
        "freeze_encoders": freeze_encoders,
        "encoder_seed": encoder_seed,
        "fusion_seed": fusion_seed,
        "fold_method": "four_training_folds_and_one_validation_fold_rotated_five_times",
        "patients": len(metadata),
        "hfref_patients": int(metadata.Label.sum()),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "folds": fold_setting_rows,
        "test_data_loaded": False,
    }
    (result_root / "run_settings.json").write_text(
        json.dumps(run_settings, indent=2) + "\n", encoding="utf-8"
    )
    return result_root


def evaluate_fusion_ensemble_on_test(
    project_dir,
    architecture="concat",
    encoder_seed=42,
    fusion_seed=42,
    batch_size=16,
):
    """Average five fusion-head probabilities, then use development-only cutoffs."""
    project_dir = Path(project_dir).resolve()
    model_root, result_root = _fusion_paths(project_dir, fusion_seed, architecture)
    development_summary_path = result_root / "development_summary.csv"
    if not development_summary_path.is_file():
        raise FileNotFoundError("Train all five fusion folds before evaluating the test set.")

    development, _ = _load_metadata(project_dir, "development")
    test, test_dir = _load_metadata(project_dir, "test")
    if set(development.Patient_ID) & set(test.Patient_ID):
        raise ValueError("Development and community test patients overlap.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        ProcessedCardioDataset(test, test_dir, modality="both"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    fold_frames = []
    expected_labels = test.Label.to_numpy(dtype=int)
    for fold in FOLDS:
        checkpoint = model_root / f"fold_{fold}" / "best_model.pt"
        settings_path = model_root / f"fold_{fold}" / "settings.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing fusion fold checkpoint: {checkpoint}")
        fold_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        model = build_fusion_model(
            fold_settings["ecg_checkpoint"],
            fold_settings["ecg_model_config"],
            fold_settings["pcg_checkpoint"],
            fold_settings["pcg_model_config"],
            architecture=fold_settings["architecture"],
            hidden_dim=fold_settings["hidden_dim"],
            dropout=fold_settings["dropout"],
            freeze_encoders=fold_settings["freeze_encoders"],
            attention_dim=fold_settings.get("attention_dim", 128),
            num_heads=fold_settings.get("num_heads", 4),
            device=device,
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        labels, probabilities = _predict_fusion(model, test_loader, device)
        if not np.array_equal(labels, expected_labels):
            raise ValueError("Community test prediction order changed.")
        frame = test[["Patient_ID", "LVEF", "Label"]].copy()
        frame["fold_model"] = fold
        frame["Probability"] = probabilities
        fold_frames.append(frame)
    fold_predictions = pd.concat(fold_frames, ignore_index=True)
    if fold_predictions.duplicated(["fold_model", "Patient_ID"]).any():
        raise ValueError("A fold model predicted a test patient more than once.")
    fold_predictions.to_csv(
        result_root / "community_test_fold_probabilities.csv", index=False
    )

    ensemble = test[["Patient_ID", "LVEF", "Label"]].copy()
    mean_probability = fold_predictions.groupby("Patient_ID").Probability.mean()
    ensemble["Probability"] = ensemble.Patient_ID.map(mean_probability)
    if not np.isfinite(ensemble.Probability).all():
        raise ValueError("Averaged test probabilities are invalid.")

    development_summary = pd.read_csv(development_summary_path)

    result_rows = []
    ensemble_prediction_frames = []

    for row in development_summary.itertuples(index=False):
        cutoff = float(row.development_cutoff)

        metrics = classification_metrics(
            test.Label,
            ensemble.Probability,
            cutoff,
        )

        model_name = f"Five-fold {architecture} fusion ensemble"

        result_rows.append({
            "model": model_name,
            "cutoff_rule": row.cutoff_rule,
            "development_cutoff": cutoff,
            **{
                f"test_{key}": value
                for key, value in metrics.items()
            },
            "result_source": "five_model_probability_average",
        })

        frame = ensemble.copy()
        frame["model"] = model_name
        frame["cutoff_rule"] = row.cutoff_rule
        frame["Prediction"] = frame.Probability.ge(cutoff)
        frame["Cutoff"] = cutoff

        ensemble_prediction_frames.append(frame)

    results = pd.DataFrame(result_rows)
    results.to_csv(result_root / "community_test_results.csv", index=False)
    pd.concat(ensemble_prediction_frames, ignore_index=True).to_csv(
        result_root / "community_test_ensemble_predictions.csv", index=False
    )
    return result_root


def compare_modalities(
    project_dir,
    ecg_seed=42,
    pcg_seed=42,
    fusion_seed=42,
    architecture="concat",
):
    """
    Ablation table:
    ECG-only vs PCG-only vs fusion development OOF metrics.
    """

    project_dir = Path(project_dir).resolve()

    modality_seeds = {
        "ecg": ecg_seed,
        "pcg": pcg_seed,
    }

    rows = []

    for modality, seed in modality_seeds.items():

        result_root = (
            project_dir
            / "analysis_outputs"
            / "five_fold_cross_validation"
            / f"{modality}_seed{seed}"
        )

        summary_path = result_root / "development_summary.csv"

        if summary_path.is_file():
            rows.append(pd.read_csv(summary_path))

    _, fusion_result_root = _fusion_paths(
        project_dir,
        fusion_seed,
        architecture,
    )

    fusion_summary_path = (
        fusion_result_root
        / "development_summary.csv"
    )

    if fusion_summary_path.is_file():
        rows.append(
            pd.read_csv(fusion_summary_path)
        )

    if not rows:
        raise FileNotFoundError(
            "No development_summary.csv found."
        )

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["cutoff_rule", "modality"])
        .reset_index(drop=True)
    )