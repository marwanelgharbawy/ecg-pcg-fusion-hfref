import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ProcessedCardioDataset(Dataset):
    def __init__(self, metadata_df, data_dir):
        self.metadata = metadata_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load the pre-saved .pt tensors
        ecg_path = os.path.join(self.data_dir, row['ECG_File'])
        pcg_path = os.path.join(self.data_dir, row['PCG_File'])
        
        ecg_tensor = torch.load(ecg_path, weights_only=True)
        pcg_tensor = torch.load(pcg_path, weights_only=True)
        
        # add channel dimension: (L) -> (1, L)
        # for PyTorch 1D Convolutional layers
        ecg_tensor = ecg_tensor.unsqueeze(0)
        pcg_tensor = pcg_tensor.unsqueeze(0)
        
        label = torch.tensor(row['Label'], dtype=torch.float32)
        
        return ecg_tensor, pcg_tensor, label

def get_dataloaders(train_df, val_df, data_dir, batch_size=16):
    train_dataset = ProcessedCardioDataset(train_df, data_dir)
    val_dataset = ProcessedCardioDataset(val_df, data_dir)
    
    # shuffle training data, keep validation sequential
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader