import torch
import torch.nn as nn
from .ecg_model import ECG_Encoder
from .pcg_model import PCG_Encoder

class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Fusing both ECG and PCG encoders
        self.ecg_net = ECG_Encoder()
        self.pcg_net = PCG_Encoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ecg, pcg):
        ecg_feat = self.ecg_net.forward_features(ecg)
        pcg_feat = self.pcg_net(pcg)
        # Feature fusion
        fused = torch.cat((ecg_feat, pcg_feat), dim=1)
        return self.classifier(fused)
