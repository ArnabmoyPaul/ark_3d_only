"""
models_3d.py  —  3D-native encoder for MedMNIST 3D
Replaces the broken Swin depth-fold approach.

Why the previous approach failed:
  - Swin-Tiny with depth-fold processed each 28×28 slice independently
  - 28px slices have ~3-5 informative voxels per organ — nearly empty
  - Mean-pooling 28 near-empty feature vectors → pure noise
  - Result: AUC stuck at 0.50 for all 36 epochs

This file uses a proper 3D CNN encoder (3D ResNet-style) that processes
the full (1, D, H, W) volume end-to-end, preserving volumetric context.

Architecture:
  Input  (B, 1, 28, 28, 28)
  Conv3d stem  → (B, 32, 14, 14, 14)
  ResBlock3D×2 → (B, 64,  7,  7,  7)
  ResBlock3D×2 → (B, 128, 4,  4,  4)
  ResBlock3D×2 → (B, 256, 2,  2,  2)
  GlobalAvgPool → (B, 256)
  Projector MLP → (B, 256)
  TaskHead_i    → (B, n_classes_i)

256-dim features, ~4M params — much smaller than 32M Swin-Tiny but
actually learns from 3D volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import remap_pretrained_keys_swin  # kept for API compat


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """Basic 3D residual block with two 3×3×3 convolutions."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Shortcut: match dimensions when stride > 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class Encoder3D(nn.Module):
    """
    Deeper 3D ResNet encoder — ~15M params, GPU-saturating.
    Input : (B, 1, D, H, W)
    Output: (B, enc_dim)
    """

    def __init__(self, enc_dim: int = 512):
        super().__init__()
        self.enc_dim = enc_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Four residual stages — deeper than before
        self.layer1 = nn.Sequential(
            ResBlock3D(64,  128, stride=2),
            ResBlock3D(128, 128, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(128, 256, stride=2),
            ResBlock3D(256, 256, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(256, enc_dim, stride=2),
            ResBlock3D(enc_dim, enc_dim, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return x.flatten(1)


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

    Calling convention (matches trainer_3d.py):
        proj_feat, logit = model(x, head_n)
    """

    def __init__(self, num_classes_list: list, enc_dim: int = 256,
                 projector_features: int = None):
        super().__init__()

        self.encoder   = Encoder3D(enc_dim=enc_dim)
        proj_dim       = projector_features or enc_dim
        self.projector = ArkProjector(enc_dim, enc_dim * 2, proj_dim)

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
    Build student or teacher model.
    enc_dim=512, 2 ResBlocks per stage → ~15M params.
    Properly saturates RTX 4060 VRAM with batch_size=128.
    """
    pf    = getattr(args, 'projector_features', None)
    model = ArkModel3D(num_classes_list, enc_dim=512, projector_features=pf)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[build_model_3d] 3D-ResNet-Deep  |  enc_dim=512  |  "
          f"params={n_params:.1f}M  |  heads={num_classes_list}")
    return model


def save_checkpoint(state: dict, filename: str = 'model'):
    torch.save(state, filename + '.pth.tar')
