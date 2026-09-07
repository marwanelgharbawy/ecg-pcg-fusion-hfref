from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
from tqdm.auto import tqdm


CHANNEL_ORDER = ("APEX", "LLSB", "LUSB", "RUSB")
CHANNEL_ORDER_STRING = "|".join(CHANNEL_ORDER)
ECG_FS = 500
PCG_FS = 4000
DURATION_SECONDS = 30
ECG_SAMPLES = ECG_FS * DURATION_SECONDS
PCG_SAMPLES = PCG_FS * DURATION_SECONDS # 120000 samples per recording


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filter along time (the last axis), independently for each channel."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Signals must be finite arrays shaped (time,) or (channels, time).")
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def _standardize(data):
    # Statistics are local to each recording/channel; no train/test statistics mix.
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    return ((data - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def preprocess_ecg(ecg_raw):
    return _standardize(butter_bandpass_filter(ecg_raw, 0.5, 40.0, ECG_FS))

# 25-400 HZ range as < 25 Hz is mostly noise and > 400 Hz is mostly noise and harmonics of the heart sounds
def preprocess_pcg(pcg_raw):
    return _standardize(butter_bandpass_filter(pcg_raw, 25.0, 400.0, PCG_FS))


def preprocess_signals(ecg_raw, pcg_raw):
    """Keep the full recording and normalize each site independently."""
    return preprocess_ecg(ecg_raw), preprocess_pcg(pcg_raw)


def _read_labels(path):
    labels = pd.read_csv(path, header=None, names=["Source_ID", "LVEF"],
                         dtype={"Source_ID": str})
    labels["LVEF"] = pd.to_numeric(labels["LVEF"], errors="raise")
    if labels.empty or labels["Source_ID"].isna().any():
        raise ValueError(f"Empty or missing IDs in {path}.")
    if not labels["LVEF"].between(0, 1).all():
        raise ValueError(f"LVEF must be a finite fraction between 0 and 1 in {path}.")
    if labels["Source_ID"].duplicated().any():
        raise ValueError(f"Duplicate label IDs in {path}.")
    labels["Label"] = (labels["LVEF"] <= 0.4).astype(int)
    return labels


def load_raw_metadata(raw_dir):
    """Return development/test recording tables, respecting their label formats.

    Fold labels use PATIENT_SITE IDs; community test labels use patient IDs and
    are expanded to the four expected sites. Test patients must be independent.
    """
    raw_dir = Path(raw_dir)
    development = []
    test = []

    def record(patient_id, channel, lvef, label, directory, fold):
        record_id = f"{patient_id}_{channel}"
        return {
            "Patient_ID": patient_id, "Record_ID": record_id, "Channel": channel,
            "LVEF": lvef, "Label": label, "Fold": fold,
            "ECG_Path": str(directory / f"{record_id}_ECG.wav"),
            "PCG_Path": str(directory / f"{record_id}_PCG.wav"),
        }

    fold_root = raw_dir / "HFrEF_5_folds"
    for fold in range(5):
        labels = _read_labels(fold_root / f"fold_{fold}_label.csv")
        for row in labels.itertuples(index=False):
            parts = row.Source_ID.rsplit("_", 1)
            if len(parts) != 2 or not parts[0] or parts[1] not in CHANNEL_ORDER:
                raise ValueError(f"Invalid recording ID: {row.Source_ID}")
            patient_id, channel = parts
            development.append(record(patient_id, channel, row.LVEF, row.Label,
                                      fold_root / f"fold_{fold}", fold))

    test_root = raw_dir / "HFrEF_community_scenario_test_set"
    for row in _read_labels(test_root / "test_set_label.csv").itertuples(index=False):
        for channel in CHANNEL_ORDER:
            test.append(record(row.Source_ID, channel, row.LVEF, row.Label,
                               test_root / "test_set", -1))

    development, test = pd.DataFrame(development), pd.DataFrame(test)
    overlap = set(development["Patient_ID"]) & set(test["Patient_ID"])
    if overlap:
        raise ValueError(f"Patients occur in development and test data: {sorted(overlap)}")
    return development, test


def build_patient_metadata(records):
    """Group four site recordings into one row per patient, or fail validation.

    Reject incomplete patients, duplicate sites, inconsistent labels/LVEF,
    patients spanning folds, and missing source files. No patient is skipped.
    """
    required = {"Patient_ID", "Record_ID", "Channel", "Label", "LVEF", "Fold",
                "ECG_Path", "PCG_Path"}
    missing = required - set(records.columns)
    if missing or records.empty:
        raise ValueError(f"Nonempty recording metadata required; missing columns: {sorted(missing)}")
    if records[list(required)].isna().any().any():
        raise ValueError("Recording metadata contains missing values.")
    records = records.copy()
    records["Patient_ID"] = records["Patient_ID"].astype(str)
    if not records["LVEF"].between(0, 1).all():
        raise ValueError("LVEF must be a fraction between 0 and 1.")
    if not records["Label"].eq((records["LVEF"] <= 0.4).astype(int)).all():
        raise ValueError("Labels must match LVEF <= 0.4.")

    patients = []
    for patient_id, group in records.groupby("Patient_ID", sort=True):
        if len(group) != 4 or set(group["Channel"]) != set(CHANNEL_ORDER):
            raise ValueError(f"Patient {patient_id} must have exactly one of each site "
                             f"{CHANNEL_ORDER}; got {group['Channel'].tolist()}.")
        for column in ("Label", "LVEF", "Fold"):
            if group[column].nunique() != 1:
                raise ValueError(f"Patient {patient_id} has inconsistent {column} values.")
        group = group.set_index("Channel").loc[list(CHANNEL_ORDER)]
        first = group.iloc[0]
        patient = {
            "Patient_ID": patient_id, "LVEF": float(first["LVEF"]),
            "Label": int(first["Label"]), "Fold": int(first["Fold"]),
            "Channel_Order": CHANNEL_ORDER_STRING, "Num_Channels": 4,
        }
        for channel, row in group.iterrows():
            if row["Record_ID"] != f"{patient_id}_{channel}":
                raise ValueError(f"Record ID does not match patient/site: {row['Record_ID']}")
            patient[f"{channel}_Record_ID"] = row["Record_ID"]
            for modality in ("ECG", "PCG"):
                path = Path(row[f"{modality}_Path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Patient {patient_id}, {channel}: {path}")
                patient[f"{channel}_{modality}_Path"] = str(path)
        patients.append(patient)
    return pd.DataFrame(patients)


def load_patient_signals(patient):
    """Stack 30-second mono WAVs as (4, time), rejecting wrong rates/durations.

    Site order does not imply temporal alignment between recordings acquired
    at different sites. No recording is silently cropped, padded, or resampled.
    """
    signals = {}
    for modality, fs, samples in (("ECG", ECG_FS, ECG_SAMPLES),
                                  ("PCG", PCG_FS, PCG_SAMPLES)):
        channels = []
        for channel in CHANNEL_ORDER:
            path = patient[f"{channel}_{modality}_Path"]
            data, actual_fs = sf.read(path, dtype="float32")
            if actual_fs != fs or data.shape != (samples,):
                raise ValueError(f"{path}: expected mono {samples} samples at {fs} Hz; "
                                 f"got shape {data.shape} at {actual_fs} Hz.")
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite samples in {path}.")
            channels.append(data)
        signals[modality] = np.stack(channels) # shape (4, time)
    return signals["ECG"], signals["PCG"]


def preprocess_dataset(patients, output_dir, show_progress=True):
    """Save one ECG and one PCG tensor per patient and a patient-level CSV."""
    if patients.empty or patients["Patient_ID"].astype(str).duplicated().any():
        raise ValueError("Expected a nonempty table with one row per patient.")
    if not patients["Channel_Order"].eq(CHANNEL_ORDER_STRING).all():
        raise ValueError(f"Channel order must be {CHANNEL_ORDER_STRING}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    for _, patient in tqdm(patients.iterrows(), total=len(patients),
                           desc="Processing patients", disable=not show_progress):
        ecg, pcg = preprocess_signals(*load_patient_signals(patient))
        patient_id = str(patient["Patient_ID"])
        ecg_filename = f"{patient_id}_ecg.pt"
        pcg_filename = f"{patient_id}_pcg.pt"
        torch.save(torch.from_numpy(ecg), output_dir / ecg_filename)
        torch.save(torch.from_numpy(pcg), output_dir / pcg_filename)
        row = {key: patient[key] for key in
               ("Patient_ID", "LVEF", "Label", "Fold", "Channel_Order", "Num_Channels")}
        row.update({f"{channel}_Record_ID": patient[f"{channel}_Record_ID"]
                    for channel in CHANNEL_ORDER})
        row.update(ECG_File=ecg_filename, PCG_File=pcg_filename,
                   ECG_Samples=ECG_SAMPLES, PCG_Samples=PCG_SAMPLES)
        processed.append(row)
    metadata = pd.DataFrame(processed)
    # Publish metadata only after every patient has been processed successfully.
    temporary_path = output_dir / "processed_metadata.csv.tmp"
    metadata.to_csv(temporary_path, index=False)
    temporary_path.replace(output_dir / "processed_metadata.csv")
    return metadata

# Not used
def segment_signals(ecg, pcg, segment_length_sec=5):
    """Time-axis segmentation."""
    ecg, pcg = np.asarray(ecg), np.asarray(pcg)
    ecg_samples = int(segment_length_sec * ECG_FS)
    pcg_samples = int(segment_length_sec * PCG_FS)
    if ecg_samples <= 0 or pcg_samples <= 0:
        raise ValueError("Segment duration must be positive.")
    if ecg.shape[-1] * PCG_FS != pcg.shape[-1] * ECG_FS:
        raise ValueError("ECG and PCG durations must match.")
    count = ecg.shape[-1] // ecg_samples
    return ([ecg[..., i * ecg_samples:(i + 1) * ecg_samples] for i in range(count)],
            [pcg[..., i * pcg_samples:(i + 1) * pcg_samples] for i in range(count)])
