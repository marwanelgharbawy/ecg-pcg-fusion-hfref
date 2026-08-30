import torch.nn as nn

class ECG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D CNN for 500Hz ECG signal
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100) 
        self.fc = nn.Linear(16 * 100, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)