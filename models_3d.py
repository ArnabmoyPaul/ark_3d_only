"""
models_3d.py  —  run09
3D ResNet encoder with aggressive dropout regularisation.

Why run08 failed on 5/6 datasets:
  - 33.8M params, ~1000 training samples per dataset
  - Model memorised training set (train loss 1.05→0.63)
  - Val loss flat at 1.0–1.2 for all 300 epochs = no generalisation
  - OrganMNIST3D (971 samples, 11 classes) still learned → architecture works
  - The other 5 binary/3-class tasks need stronger regularisation

Fixes:
  1. enc_dim 512→256  (~4M params, 8× fewer than run08)
  2. Dropout(p=0.4) after every ResBlock activation
  3. Dropout(p=0.5) on the final feature vector before heads
  4. weight_decay 0.1 in main_3d.py (was 0.05)
  5. aug probability 0.5 in dataloader_3d.py (was 0.3)
"""

import torch
import torch.nn as nn

from utils import remap_pretrained_keys_swin   # kept for API compat


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """3D residual block with mild dropout after second conv."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # Regular Dropout (not Dropout3d) — drops individual voxels not channels
        # Dropout3d drops entire feature maps which is too aggressive on 28^3 vols
        self.drop  = nn.Dropout(p=dropout)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))   # dropout after second BN
        out = out + self.shortcut(x)
        return self.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    """
    3D ResNet encoder — enc_dim=384, mild dropout.
    Dropout3d was too aggressive (kills entire feature maps on 28^3 volumes).
    Regular Dropout at p=0.15-0.25 gives regularisation without strangling gradients.
    """

    def __init__(self, enc_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.enc_dim = enc_dim

        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResBlock3D(32,  64,  stride=2, dropout=dropout * 0.5),   # p=0.10
            ResBlock3D(64,  64,  stride=1, dropout=dropout * 0.5),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(64,  128, stride=2, dropout=dropout * 0.75),  # p=0.15
            ResBlock3D(128, 128, stride=1, dropout=dropout * 0.75),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(128, enc_dim, stride=2, dropout=dropout),     # p=0.20
            ResBlock3D(enc_dim, enc_dim, stride=1, dropout=dropout),
        )

        self.pool         = nn.AdaptiveAvgPool3d(1)
        self.feature_drop = nn.Dropout(p=0.3)   # final feature dropout

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.feature_drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Projection head
# ─────────────────────────────────────────────────────────────────────────────

class ArkProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = None, out_dim: int = None):
        super().__init__()
        h = hidden_dim or in_dim * 2
        o = out_dim    or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Linear(h, o),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class ArkModel3D(nn.Module):
    """
    3D encoder + projector + per-dataset classification heads.
    Calling convention: proj_feat, logit = model(x, head_n)
    """

    def __init__(self, num_classes_list: list,
                 enc_dim: int = 256,
                 dropout: float = 0.4,
                 projector_features: int = None):
        super().__init__()

        self.encoder    = Encoder3D(enc_dim=enc_dim, dropout=dropout)
        proj_dim        = projector_features or enc_dim
        self.projector  = ArkProjector(enc_dim, enc_dim * 2, proj_dim)

        self.omni_heads = nn.ModuleList([
            nn.Linear(enc_dim, nc) for nc in num_classes_list
        ])
        for h in self.omni_heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.constant_(h.bias, 0.0)

        self._enc_dim  = enc_dim
        self._proj_dim = proj_dim

    def forward(self, x: torch.Tensor, head_n: int = None):
        enc  = self.encoder(x)
        proj = self.projector(enc)
        if head_n is not None:
            return proj, self.omni_heads[head_n](enc)
        return [h(enc) for h in self.omni_heads]

    def generate_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model_3d(args, num_classes_list: list) -> ArkModel3D:
    """
    enc_dim=384, regular Dropout(0.2) in ResBlocks, Dropout(0.3) on features.
    ~11M params — enough capacity to learn, enough dropout to not memorise.
    """
    pf      = getattr(args, 'projector_features', None)
    dropout = getattr(args, 'dropout', 0.2)
    model   = ArkModel3D(num_classes_list,
                         enc_dim=384,
                         dropout=dropout,
                         projector_features=pf)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[build_model_3d] 3D-ResNet+Dropout  |  enc_dim=384  |  "
          f"dropout={dropout}  |  params={n_params:.1f}M  |  "
          f"heads={num_classes_list}")
    return model


def save_checkpoint(state: dict, filename: str = 'model'):
    torch.save(state, filename + '.pth.tar')
