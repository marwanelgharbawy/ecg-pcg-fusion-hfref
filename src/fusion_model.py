"""Simple concatenation fusion of the pretrained ECG and PCG encoders.

Each encoder's own classifier head is discarded; only `forward_features` is
used. The two patient embeddings are concatenated and passed through a small
MLP classifier. Encoders can be frozen (feature-extraction fusion) or left
trainable (end-to-end fine-tuning fusion) via `freeze_encoders`.
"""

import torch
import torch.nn as nn

from .ecg_model import ECG_Encoder
from .pcg_model import PCG_Encoder


class ConcatFusionModel(nn.Module):
    # forward returns one logit per patient, same convention as ECG_Encoder
    # and PCG_Encoder.

    def __init__(self, ecg_encoder, pcg_encoder, hidden_dim=128, dropout=0.3,
                freeze_encoders=True):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.pcg_encoder = pcg_encoder
        self.freeze_encoders = freeze_encoders
        if freeze_encoders:
            for encoder in (self.ecg_encoder, self.pcg_encoder):
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
        combined_dim = self.ecg_encoder.feature_dim + self.pcg_encoder.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        # Keep frozen encoders in eval mode (frozen BatchNorm running stats,
        # disabled Dropout) even while the fusion head is in training mode.
        super().train(mode)
        if self.freeze_encoders:
            self.ecg_encoder.eval()
            self.pcg_encoder.eval()
        return self

    def forward_features(self, ecg, pcg):
        if self.freeze_encoders:
            with torch.no_grad():
                ecg_features = self.ecg_encoder.forward_features(ecg)
                pcg_features = self.pcg_encoder.forward_features(pcg)
        else:
            ecg_features = self.ecg_encoder.forward_features(ecg)
            pcg_features = self.pcg_encoder.forward_features(pcg)
        return torch.cat((ecg_features, pcg_features), dim=1)

    def forward(self, ecg, pcg):
        return self.classifier(self.forward_features(ecg, pcg))

class AttentionFusionModel(nn.Module):
    """
    Attention-based fusion of pretrained ECG and PCG patient embeddings.

    ECG embedding -> token 1
    PCG embedding -> token 2

    Self-attention learns how the two modality representations interact
    before final HFrEF classification.
    """

    def __init__(
        self,
        ecg_encoder,
        pcg_encoder,
        attention_dim=128,
        num_heads=4,
        hidden_dim=128,
        dropout=0.3,
        freeze_encoders=True,
    ):
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")

        self.ecg_encoder = ecg_encoder
        self.pcg_encoder = pcg_encoder
        self.freeze_encoders = freeze_encoders

        if freeze_encoders:
            for encoder in (self.ecg_encoder, self.pcg_encoder):
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)

        # Separate learned projections let ECG and PCG map into the same
        # attention space even if their encoder feature dimensions differ.
        self.ecg_projection = nn.Linear(
            self.ecg_encoder.feature_dim,
            attention_dim,
        )

        self.pcg_projection = nn.Linear(
            self.pcg_encoder.feature_dim,
            attention_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(attention_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attention_dim),
        )

        self.norm2 = nn.LayerNorm(attention_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * attention_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_encoders:
            self.ecg_encoder.eval()
            self.pcg_encoder.eval()

        return self

    def _encoder_features(self, ecg, pcg):

        if self.freeze_encoders:
            with torch.no_grad():
                ecg_features = self.ecg_encoder.forward_features(ecg)
                pcg_features = self.pcg_encoder.forward_features(pcg)
        else:
            ecg_features = self.ecg_encoder.forward_features(ecg)
            pcg_features = self.pcg_encoder.forward_features(pcg)

        return ecg_features, pcg_features

    def forward_features(self, ecg, pcg):

        ecg_features, pcg_features = self._encoder_features(ecg, pcg)

        ecg_token = self.ecg_projection(ecg_features)
        pcg_token = self.pcg_projection(pcg_features)

        # [batch, 2 modalities, attention_dim]
        tokens = torch.stack(
            (ecg_token, pcg_token),
            dim=1,
        )

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        x = self.norm1(tokens + attended)

        ff = self.feed_forward(x)

        x = self.norm2(x + ff)

        # Keep information from both modality tokens.
        return x.flatten(start_dim=1)

    def forward_with_attention(self, ecg, pcg):

        ecg_features, pcg_features = self._encoder_features(ecg, pcg)

        tokens = torch.stack(
            (
                self.ecg_projection(ecg_features),
                self.pcg_projection(pcg_features),
            ),
            dim=1,
        )

        attended, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        x = self.norm1(tokens + attended)
        x = self.norm2(x + self.feed_forward(x))

        logits = self.classifier(x.flatten(start_dim=1))

        return logits, attention_weights

    def forward(self, ecg, pcg):

        features = self.forward_features(ecg, pcg)

        return self.classifier(features)
    
    
def load_pretrained_encoder(encoder_class, model_config, checkpoint_path, device="cpu"):
    """Instantiate an encoder and load a saved fold checkpoint's full state dict.

    The encoder's own `classifier` layer is loaded too but is never called by
    the fusion model afterwards; only `forward_features` is used.
    """
    encoder = encoder_class(**model_config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state_dict)
    return encoder.to(device)


def build_fusion_model(
    ecg_checkpoint_path,
    ecg_model_config,
    pcg_checkpoint_path,
    pcg_model_config,
    architecture="concat",
    hidden_dim=128,
    dropout=0.3,
    freeze_encoders=True,
    attention_dim=128,
    num_heads=4,
    device="cpu",
):
    """
    Build a pretrained multimodal fusion model.

    architecture:
        "concat"     -> simple feature concatenation
        "attention"  -> two-token ECG/PCG self-attention
    """

    device = torch.device(device)

    ecg_encoder = load_pretrained_encoder(
        ECG_Encoder,
        ecg_model_config,
        ecg_checkpoint_path,
        device,
    )

    pcg_encoder = load_pretrained_encoder(
        PCG_Encoder,
        pcg_model_config,
        pcg_checkpoint_path,
        device,
    )

    architecture = architecture.lower()

    if architecture == "concat":

        model = ConcatFusionModel(
            ecg_encoder=ecg_encoder,
            pcg_encoder=pcg_encoder,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_encoders=freeze_encoders,
        )

    elif architecture == "attention":

        model = AttentionFusionModel(
            ecg_encoder=ecg_encoder,
            pcg_encoder=pcg_encoder,
            attention_dim=attention_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_encoders=freeze_encoders,
        )

    else:
        raise ValueError(
            "architecture must be 'concat' or 'attention'."
        )

    return model.to(device)