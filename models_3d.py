"""
models_3d.py  —  run10
3D ResNet with fixes for the 3 confirmed problems:

Fix A — Reduced striding (CRITICAL)
  Previous: 28→14→7→4→2 (4× stride-2 ops) — feature maps collapse to 2³
  Now:      28→28→14→7→7 (only 2× stride-2) — preserves spatial detail
  Why: VesselMNIST3D (tiny aneurysm) and SynapseMNIST3D (EM subtleties)
  need spatial resolution. Can't detect a 3-voxel bulge from a 2×2×2 map.

Fix B — GroupNorm replaces BatchNorm3d (IMPORTANT)
  Previous: 16× BatchNorm3d — running stats corrupted by 6 different
            domain distributions in round-robin multi-task training
  Now:      GroupNorm(8, channels) — per-sample, no running stats,
            immune to multi-task distribution mixing

Fix C — Loss scale handled in engine_3d.py (not here)
  Organ CE loss = 2.40, binary BCE = 0.69 → 3.5× gradient imbalance
  Fixed by normalising each loss by ln(n_classes) before backward
"""

import math
import torch
import torch.nn as nn

from utils import remap_pretrained_keys_swin   # kept for API compat


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks — GroupNorm + regular Dropout
# ─────────────────────────────────────────────────────────────────────────────

def _gn(n_ch: int) -> nn.GroupNorm:
    """GroupNorm with 8 groups (works for ch >= 8)."""
    n_groups = min(8, n_ch)
    # GroupNorm requires num_channels divisible by num_groups
    while n_ch % n_groups != 0 and n_groups > 1:
        n_groups -= 1
    return nn.GroupNorm(n_groups, n_ch)


class ResBlock3D(nn.Module):
    """
    3D residual block.
    GroupNorm instead of BatchNorm — no running stats, multi-task safe.
    Regular Dropout (not Dropout3d) — drops voxels not whole channels.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.gn1   = _gn(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.gn2   = _gn(out_ch)

        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                _gn(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.gn2(self.conv2(out)))
        out = out + self.shortcut(x)
        return self.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder — reduced striding to preserve spatial resolution
# ─────────────────────────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    """
    Spatial dimension trace (28³ input):
      Stem   (stride=1): 28→28  (no spatial reduction — keep all information)
      Layer1 (stride=1): 28→28  channels 32→64
      Layer2 (stride=2): 28→14  channels 64→128   ← only 1st downsampling
      Layer3 (stride=2): 14→7   channels 128→enc  ← only 2nd downsampling
      Pool             :  7→1   global avg

    Result: 7³=343 voxels pooled vs previous 2³=8 voxels.
    43× more spatial information preserved before pooling.
    Critical for detecting small structures (nodules, aneurysms, synapses).
    """

    def __init__(self, enc_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.enc_dim = enc_dim

        # Stem: stride=1, no spatial reduction
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            _gn(32),
            nn.ReLU(inplace=True),
        )

        # Layer1: stride=1, build features at full resolution
        self.layer1 = nn.Sequential(
            ResBlock3D(32,  64,  stride=1, dropout=dropout * 0.5),
            ResBlock3D(64,  64,  stride=1, dropout=dropout * 0.5),
        )

        # Layer2: stride=2 (28→14), first downsampling
        self.layer2 = nn.Sequential(
            ResBlock3D(64,  128, stride=2, dropout=dropout * 0.75),
            ResBlock3D(128, 128, stride=1, dropout=dropout * 0.75),
        )

        # Layer3: stride=2 (14→7), second downsampling
        self.layer3 = nn.Sequential(
            ResBlock3D(128, enc_dim, stride=2, dropout=dropout),
            ResBlock3D(enc_dim, enc_dim, stride=1, dropout=dropout),
        )

        self.pool         = nn.AdaptiveAvgPool3d(1)
        self.feature_drop = nn.Dropout(p=0.3)

        # Kaiming init for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

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
            nn.LayerNorm(h),      # LayerNorm not BatchNorm1d — multi-task safe
            nn.ReLU(inplace=True),
            nn.Linear(h, o),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class ArkModel3D(nn.Module):
    def __init__(self, num_classes_list: list,
                 enc_dim: int = 384,
                 dropout: float = 0.2,
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
    pf      = getattr(args, 'projector_features', None)
    dropout = getattr(args, 'dropout', 0.2)
    model   = ArkModel3D(num_classes_list,
                         enc_dim=384,
                         dropout=dropout,
                         projector_features=pf)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Verify spatial dimensions
    with torch.no_grad():
        x = torch.zeros(1, 1, 28, 28, 28)
        e = model.encoder
        x = e.stem(x); s1 = x.shape[2]
        x = e.layer1(x); s2 = x.shape[2]
        x = e.layer2(x); s3 = x.shape[2]
        x = e.layer3(x); s4 = x.shape[2]

    print(f"[build_model_3d] 3D-ResNet+GN  |  enc_dim=384  |  "
          f"dropout={dropout}  |  params={n_params:.1f}M  |  "
          f"spatial: 28→{s1}→{s2}→{s3}→{s4}→pool  |  "
          f"heads={num_classes_list}")
    return model


def save_checkpoint(state: dict, filename: str = 'model'):
    torch.save(state, filename + '.pth.tar')
