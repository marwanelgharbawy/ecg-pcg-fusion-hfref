import torch
import torch.nn as nn

from .ecg_model import ResidualBlock1D


class PCG_Encoder(nn.Module):
    # Input: (batch, 4, 120000), ordered APEX, LLSB, LUSB, RUSB
    # PCG is sampled at 4000 Hz (vs ECG's 500 Hz), so the same 30s recording
    # 30 x 4000 = 120000 samples per channel, while ECG's 30s recording is 30 x 500 = 15000 samples per channel.
    # is 8x longer in raw samples. The stem downsamples more aggressively
    # than the ECG stem (total /16 here vs /4 for ECG) so that after the
    # same 4 residual blocks, the final sequence length matches ECG's (~469).
    # forward returns one logit per patient

    def __init__(self, in_channels=4, feature_dim=64, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=4, padding=2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(16, 16),
            ResidualBlock1D(16, 32, stride=2),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64, stride=2),
        )
        # avg_pool answers how strongly is this feature present overall across the recording 
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # max_pool answers What is the strongest occurrence of this feature anywhere in the recording?
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.embedding = nn.Sequential(
            # from 128 (64 from avg_pool + 64 from max_pool) to feature_dim (64)
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, 1)

    # returns 128-dim pcg embedding for each patient
    def forward_features(self, x):
        if x.ndim != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected (batch, {self.in_channels}, time), got {tuple(x.shape)}.")
        x = self.blocks(self.stem(x))
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1).flatten(1)
        return self.embedding(x)

    # returns a single logit for each patient
    def forward(self, x):
        return self.classifier(self.forward_features(x))