import torch.nn as nn

class PCG_Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # 1D CNN for 4000Hz PCG signal
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=31, stride=4, padding=15)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(4)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)
        self.fc = nn.Linear(16 * 100, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
