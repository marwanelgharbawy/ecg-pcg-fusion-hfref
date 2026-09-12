"""Five-fold ECG and PCG training with one prediction per patient."""

from __future__ import annotations

import json
import random
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
from .ecg_model import ECG_Encoder
from .pcg_model import PCG_Encoder
from .preprocessing import (
    CHANNEL_ORDER,
    CHANNEL_ORDER_STRING,
    ECG_FS,
    ECG_SAMPLES,
    PCG_FS,
    PCG_SAMPLES,
)
from .train import FocalLoss


FOLDS = tuple(range(5))
CUTOFF_RULES = ("maximum_development_f1", "target_90pct_development_specificity")

MODALITY_SETTINGS = {
    "ecg": {
        "model_class": ECG_Encoder,
        "model_config": {"in_channels": 4, "feature_dim": 128, "dropout": 0.3},
        "samples": ECG_SAMPLES,
        "sample_rate": ECG_FS,
        "weight_decay": 1e-4,
        "training_crop_samples": None,
        "old_checkpoint": "best_ecg_4ch_fold0.pt",
        "old_settings": "best_ecg_4ch_fold0.json",
    },
    "pcg": {
        "model_class": PCG_Encoder,
        "model_config": {"in_channels": 4, "feature_dim": 128, "dropout": 0.5},
        "samples": PCG_SAMPLES,
        "sample_rate": PCG_FS,
        "weight_decay": 1e-3,
        "training_crop_samples": 15 * PCG_FS,
        "old_checkpoint": "best_pcg_4ch_fold0.pt",
        "old_settings": "best_pcg_4ch_fold0.json",
    },
}

# The old PCG checkpoint is not in the repository. These values come from the
# saved output in notebooks/3_pcg_baseline.ipynb and are marked as a reference.
OLD_PCG_TEST_REFERENCE = {
    "model": "Original notebook 3.1 single model",
    "cutoff_rule": "original_saved_cutoff",
    "development_cutoff": 0.502835750579834,
    "test_patients": 120,
    "test_hfref_patients": 5,
    "test_threshold": 0.502835750579834,
    "test_auprc": 0.1623,
    "test_auroc": 0.8748,
    "test_recall": 1.0,
    "test_specificity": 72 / 115,
    "test_balanced_accuracy": (1.0 + 72 / 115) / 2,
    "test_precision": 5 / 48,
    "test_f1": 10 / 53,
    "test_brier": float("nan"),
    "test_log_loss": float("nan"),
    "test_calibration_error": float("nan"),
    "test_tn": 72,
    "test_fp": 43,
    "test_fn": 0,
    "test_tp": 5,
    "result_source": "saved_notebook_output_checkpoint_unavailable",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _paths(project_dir: Path, modality: str, seed: int) -> tuple[Path, Path]:
    model_dir = (
        project_dir / "models" / "five_fold_cross_validation"
        / f"{modality}_seed{seed}"
    )
    result_dir = (
        project_dir / "analysis_outputs" / "five_fold_cross_validation"
        / f"{modality}_seed{seed}"
    )
    return model_dir, result_dir


def _load_metadata(project_dir: Path, split: str) -> tuple[pd.DataFrame, Path]:
    data_dir = project_dir / "data" / "processed" / "patient_4ch" / split
    metadata = pd.read_csv(
        data_dir / "processed_metadata.csv", dtype={"Patient_ID": str}
    )
    expected = 500 if split == "development" else 120
    if len(metadata) != expected or metadata.Patient_ID.duplicated().any():
        raise ValueError(f"Expected {expected} unique {split} patients.")
    if not metadata.Channel_Order.eq(CHANNEL_ORDER_STRING).all():
        raise ValueError(f"Expected site order {CHANNEL_ORDER_STRING}.")
    if not metadata.Label.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    expected_labels = metadata.LVEF.le(0.40).astype(int)
    if not np.array_equal(metadata.Label.to_numpy(), expected_labels.to_numpy()):
        raise ValueError("Labels do not match the LVEF <= 40% rule.")
    if split == "development" and set(metadata.Fold.unique()) != set(FOLDS):
        raise ValueError("Development metadata must contain folds 0 through 4.")
    if split == "test" and not metadata.Fold.eq(-1).all():
        raise ValueError("Community test patients must have fold -1.")
    return metadata, data_dir


def inspect_five_fold_inputs(project_dir, modality=None) -> pd.DataFrame:
    """Run metadata and one-tensor checks without training or loading test data."""
    project_dir = Path(project_dir).resolve()
    metadata, data_dir = _load_metadata(project_dir, "development")
    fold_counts = metadata.groupby("Fold").Label.agg(["size", "sum"])
    if not fold_counts["size"].eq(100).all() or not fold_counts["sum"].eq(12).all():
        raise ValueError("Expected 100 patients and 12 HFrEF patients in every fold.")
    if modality is not None:
        modality = modality.lower()
        if modality not in MODALITY_SETTINGS:
            raise ValueError("modality must be ecg or pcg.")
        modalities = (modality,)
    else:
        modalities = tuple(MODALITY_SETTINGS)
    rows = []
    for modality in modalities:
        settings = MODALITY_SETTINGS[modality]
        dataset = ProcessedCardioDataset(metadata.iloc[[0]], data_dir, modality=modality)
        signal, label = dataset[0]
        expected_shape = (4, settings["samples"])
        if tuple(signal.shape) != expected_shape or not torch.isfinite(signal).all():
            raise ValueError(f"Unexpected {modality.upper()} tensor.")
        model = settings["model_class"](**settings["model_config"])
        previous_mode = model.training
        model.eval()
        with torch.no_grad():
            output = model(signal.unsqueeze(0))
        model.train(previous_mode)
        if tuple(output.shape) != (1, 1) or not torch.isfinite(output).all():
            raise ValueError(f"Unexpected {modality.upper()} model output.")
        rows.append({
            "modality": modality.upper(),
            "patients": len(metadata),
            "hfref_patients": int(metadata.Label.sum()),
            "fold_sizes": str(metadata.groupby("Fold").size().to_dict()),
            "input_shape": str(expected_shape),
            "model_output_shape": str(tuple(output.shape)),
            "training_crop_seconds": (
                settings["training_crop_samples"] / settings["sample_rate"]
                if settings["training_crop_samples"] else 30
            ),
        })
    return pd.DataFrame(rows)


def _make_loaders(train_df, validation_df, data_dir, modality, seed, batch_size):
    train_dataset = ProcessedCardioDataset(train_df, data_dir, modality=modality)
    validation_dataset = ProcessedCardioDataset(
        validation_df, data_dir, modality=modality
    )
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


def _crop_training_batch(values, crop_samples, generator):
    if crop_samples is None:
        return values
    if values.shape[-1] < crop_samples:
        raise ValueError("Training crop is longer than the recording.")
    latest_start = values.shape[-1] - crop_samples
    starts = torch.randint(
        0, latest_start + 1, (values.shape[0],), generator=generator
    )
    return torch.stack([
        values[index, :, int(start):int(start) + crop_samples]
        for index, start in enumerate(starts)
    ])


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for values, batch_labels in loader:
        logits = model(values.to(device, non_blocking=True))
        if tuple(logits.shape) != (len(batch_labels), 1):
            raise ValueError("Expected one model output per patient.")
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced an invalid output.")
        labels.extend(batch_labels.numpy().astype(int))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def _train_one_fold(
    project_dir,
    modality,
    seed,
    validation_fold,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    model_root,
):
    settings = MODALITY_SETTINGS[modality]
    metadata, data_dir = _load_metadata(project_dir, "development")
    train_df = metadata[metadata.Fold.ne(validation_fold)].reset_index(drop=True)
    validation_df = metadata[metadata.Fold.eq(validation_fold)].reset_index(drop=True)
    if len(train_df) != 400 or len(validation_df) != 100:
        raise ValueError("Each run requires 400 training and 100 validation patients.")
    if int(validation_df.Label.sum()) != 12:
        raise ValueError("Expected 12 HFrEF patients in every validation fold.")

    fold_seed = seed + validation_fold
    _set_seed(fold_seed)
    train_loader, validation_loader = _make_loaders(
        train_df, validation_df, data_dir, modality, fold_seed, batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = settings["model_class"](**settings["model_config"]).to(device)
    sample, _ = next(iter(validation_loader))
    if tuple(sample.shape[1:]) != (4, settings["samples"]):
        raise ValueError(f"Unexpected {modality.upper()} input shape.")
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        checked_output = model(sample[:1].to(device))
    model.train(previous_mode)
    if tuple(checked_output.shape) != (1, 1):
        raise ValueError("Model shape check failed.")

    focal_alpha = float(train_df.Label.eq(0).mean())
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    crop_generator = torch.Generator().manual_seed(fold_seed + 100_000)
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
        for values, labels in train_loader:
            values = _crop_training_batch(
                values, settings["training_crop_samples"], crop_generator
            ).to(device, non_blocking=True)
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

        validation_labels, validation_probabilities = _predict(
            model, validation_loader, device
        )
        validation_auroc = roc_auc_score(
            validation_labels, validation_probabilities
        )
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
            f"{modality.upper()} fold {validation_fold} | epoch {epoch} | "
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
        raise RuntimeError("Training did not save a checkpoint.")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    validation_labels, validation_probabilities = _predict(
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
        "modality": modality,
        "base_seed": seed,
        "fold_seed": fold_seed,
        "validation_fold": validation_fold,
        "training_folds": [fold for fold in FOLDS if fold != validation_fold],
        "training_patients": len(train_df),
        "validation_patients": len(validation_df),
        "training_hfref_patients": int(train_df.Label.sum()),
        "validation_hfref_patients": int(validation_df.Label.sum()),
        "training_patient_ids": train_df.Patient_ID.tolist(),
        "validation_patient_ids": validation_df.Patient_ID.tolist(),
        "model_config": settings["model_config"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": settings["weight_decay"],
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
        "training_crop_seconds": (
            settings["training_crop_samples"] / settings["sample_rate"]
            if settings["training_crop_samples"] else None
        ),
        "validation_input_seconds": 30,
        "site_order": list(CHANNEL_ORDER),
        "shape_check_mode": "evaluation_without_gradients",
    }
    (checkpoint_dir / "settings.json").write_text(
        json.dumps(fold_settings, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, history_frame, fold_settings


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


def train_five_fold_models(
    project_dir,
    modality,
    seed=42,
    batch_size=16,
    learning_rate=1e-3,
    max_epochs=50,
    patience=10,
):
    """Train five models and save one out-of-fold prediction per patient."""
    project_dir = Path(project_dir).resolve()
    modality = modality.lower()
    if modality not in MODALITY_SETTINGS:
        raise ValueError("modality must be ecg or pcg.")
    model_root, result_root = _paths(project_dir, modality, seed)
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
        predictions, _, fold_settings = _train_one_fold(
            project_dir=project_dir,
            modality=modality,
            seed=seed,
            validation_fold=validation_fold,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            model_root=model_root,
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
            "modality": modality,
            "seed": seed,
            "cutoff_rule": rule,
            "development_cutoff": cutoff,
            "cutoff_selection_specificity": achieved_specificity,
            **metrics,
        })
        oof[f"Prediction_{rule}"] = oof.Probability.ge(cutoff)
        for fold, frame in oof.groupby("Fold"):
            fold_metrics = classification_metrics(
                frame.Label, frame.Probability, cutoff
            )
            fold_rows.append({
                "modality": modality,
                "seed": seed,
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
        "modality": modality,
        "seed": seed,
        "fold_method": "four_training_folds_and_one_validation_fold_rotated_five_times",
        "patients": len(metadata),
        "hfref_patients": int(metadata.Label.sum()),
        "model_settings": MODALITY_SETTINGS[modality]["model_config"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": MODALITY_SETTINGS[modality]["weight_decay"],
        "max_epochs": max_epochs,
        "patience": patience,
        "folds": fold_setting_rows,
        "test_data_loaded": False,
    }
    (result_root / "run_settings.json").write_text(
        json.dumps(run_settings, indent=2) + "\n", encoding="utf-8"
    )
    return result_root


def _old_ecg_test_result(project_dir, test, test_dir, device, batch_size):
    settings_path = project_dir / "models" / "best_ecg_4ch_fold0.json"
    checkpoint_path = project_dir / "models" / "best_ecg_4ch_fold0.pt"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    model = ECG_Encoder(**settings["model_config"]).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    loader = DataLoader(
        ProcessedCardioDataset(test, test_dir, modality="ecg"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    labels, probabilities = _predict(model, loader, device)
    metrics = classification_metrics(labels, probabilities, settings["threshold"])
    row = {
        "model": "Original notebook 2.1 single model",
        "cutoff_rule": "original_saved_cutoff",
        "development_cutoff": settings["threshold"],
        **{f"test_{key}": value for key, value in metrics.items()},
        "result_source": "replayed_saved_checkpoint",
    }
    predictions = test[["Patient_ID", "LVEF", "Label"]].copy()
    predictions["model"] = row["model"]
    predictions["Probability"] = probabilities
    predictions["Prediction"] = probabilities >= settings["threshold"]
    predictions["Cutoff"] = settings["threshold"]
    return row, predictions


def _old_pcg_test_result(project_dir, test, test_dir, device, batch_size):
    checkpoint = project_dir / "models" / "best_pcg_4ch_fold0.pt"
    settings_path = project_dir / "models" / "best_pcg_4ch_fold0.json"
    if not checkpoint.is_file() or not settings_path.is_file():
        return OLD_PCG_TEST_REFERENCE.copy(), None
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    model = PCG_Encoder(**settings["model_config"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(
        ProcessedCardioDataset(test, test_dir, modality="pcg"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    labels, probabilities = _predict(model, loader, device)
    metrics = classification_metrics(labels, probabilities, settings["threshold"])
    row = {
        "model": "Original notebook 3.1 single model",
        "cutoff_rule": "original_saved_cutoff",
        "development_cutoff": settings["threshold"],
        **{f"test_{key}": value for key, value in metrics.items()},
        "result_source": "replayed_saved_checkpoint",
    }
    predictions = test[["Patient_ID", "LVEF", "Label"]].copy()
    predictions["model"] = row["model"]
    predictions["Probability"] = probabilities
    predictions["Prediction"] = probabilities >= settings["threshold"]
    predictions["Cutoff"] = settings["threshold"]
    return row, predictions


def evaluate_five_model_ensemble_on_test(
    project_dir, modality, seed=42, batch_size=16
):
    """Average five model probabilities, then use development-only cutoffs."""
    project_dir = Path(project_dir).resolve()
    modality = modality.lower()
    if modality not in MODALITY_SETTINGS:
        raise ValueError("modality must be ecg or pcg.")
    model_root, result_root = _paths(project_dir, modality, seed)
    development_summary_path = result_root / "development_summary.csv"
    if not development_summary_path.is_file():
        raise FileNotFoundError("Train all five models before evaluating the test set.")
    test_results_path = result_root / "community_test_results.csv"

    development, _ = _load_metadata(project_dir, "development")
    test, test_dir = _load_metadata(project_dir, "test")
    if set(development.Patient_ID) & set(test.Patient_ID):
        raise ValueError("Development and community test patients overlap.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        ProcessedCardioDataset(test, test_dir, modality=modality),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    settings = MODALITY_SETTINGS[modality]
    fold_frames = []
    expected_labels = test.Label.to_numpy(dtype=int)
    for fold in FOLDS:
        checkpoint = model_root / f"fold_{fold}" / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")
        model = settings["model_class"](**settings["model_config"]).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        labels, probabilities = _predict(model, test_loader, device)
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
        metrics = classification_metrics(test.Label, ensemble.Probability, cutoff)
        model_name = f"Five-model {modality.upper()} ensemble"
        result_rows.append({
            "model": model_name,
            "cutoff_rule": row.cutoff_rule,
            "development_cutoff": cutoff,
            **{f"test_{key}": value for key, value in metrics.items()},
            "result_source": "five_model_probability_average",
        })
        frame = ensemble.copy()
        frame["model"] = model_name
        frame["cutoff_rule"] = row.cutoff_rule
        frame["Prediction"] = frame.Probability.ge(cutoff)
        frame["Cutoff"] = cutoff
        ensemble_prediction_frames.append(frame)

    if modality == "ecg":
        old_row, old_predictions = _old_ecg_test_result(
            project_dir, test, test_dir, device, batch_size
        )
    else:
        old_row, old_predictions = _old_pcg_test_result(
            project_dir, test, test_dir, device, batch_size
        )
    result_rows.append(old_row)
    if old_predictions is not None:
        old_predictions.to_csv(
            result_root / "original_model_community_test_predictions.csv",
            index=False,
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(test_results_path, index=False)
    pd.concat(ensemble_prediction_frames, ignore_index=True).to_csv(
        result_root / "community_test_ensemble_predictions.csv", index=False
    )
    return result_root
