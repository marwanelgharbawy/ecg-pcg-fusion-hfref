import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.skip(x))


class ECG_Encoder(nn.Module):
    # Input: (batch, 4, 15000), ordered APEX, LLSB, LUSB, RUSB.
    # forward returns one logit per patient

    def __init__(self, in_channels=4, feature_dim=128, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.embedding = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def forward_features(self, x):
        if x.ndim != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected (batch, {self.in_channels}, time), got {tuple(x.shape)}.")
        x = self.blocks(self.stem(x))
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1).flatten(1)
        return self.embedding(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))
