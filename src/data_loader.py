from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import CHANNEL_ORDER_STRING, ECG_SAMPLES, PCG_SAMPLES


class ProcessedCardioDataset(Dataset):
    def __init__(self, metadata_df, data_dir, modality="both"):
        if modality not in ("ecg", "pcg", "both"):
            raise ValueError("modality must be 'ecg', 'pcg', or 'both'.")
        required = {"Patient_ID", "Label", "Channel_Order"}
        if modality in ("ecg", "both"):
            required.add("ECG_File")
        if modality in ("pcg", "both"):
            required.add("PCG_File")
        if not required.issubset(metadata_df.columns):
            raise ValueError("Patient-level metadata required. Run 1_preprocessing.ipynb first.")
        if metadata_df.empty or metadata_df[list(required)].isna().any().any():
            raise ValueError("Metadata must be nonempty and contain no missing required values.")
        if metadata_df["Patient_ID"].astype(str).duplicated().any():
            raise ValueError("Metadata must contain exactly one row per patient.")
        if not metadata_df["Channel_Order"].eq(CHANNEL_ORDER_STRING).all():
            raise ValueError(f"Expected channel order {CHANNEL_ORDER_STRING}.")
        if not metadata_df["Label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary (0 or 1).")
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.modality = modality

    def __len__(self):
        return len(self.metadata)

    def _load(self, filename, samples):
        path = self.data_dir / filename
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (4, samples):
            raise ValueError(f"{path}: expected a (4, {samples}) patient tensor. "
                             "Regenerate data with 1_preprocessing.ipynb.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path}: tensor contains nonfinite values.")
        return tensor.float()

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        label = torch.tensor(row["Label"], dtype=torch.float32)
        if self.modality == "ecg":
            return self._load(row["ECG_File"], ECG_SAMPLES), label
        if self.modality == "pcg":
            return self._load(row["PCG_File"], PCG_SAMPLES), label
        return (self._load(row["ECG_File"], ECG_SAMPLES),
                self._load(row["PCG_File"], PCG_SAMPLES), label)


def get_dataloaders(train_df, val_df, data_dir, batch_size=16, modality="both",
                    num_workers=0, seed=42):
    train_dataset = ProcessedCardioDataset(train_df, data_dir, modality)
    val_dataset = ProcessedCardioDataset(val_df, data_dir, modality)
    overlap = set(train_dataset.metadata["Patient_ID"].astype(str)) & set(
        val_dataset.metadata["Patient_ID"].astype(str))
    if overlap:
        raise ValueError(f"Training and validation patients overlap: {sorted(overlap)}")
    options = dict(batch_size=batch_size, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train_dataset, shuffle=True, generator=generator, **options),
            DataLoader(val_dataset, shuffle=False, **options))
