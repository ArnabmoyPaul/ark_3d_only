"""
models_3d.py
────────────
Ark+ backbone for 3D-only MedMNIST pretraining.

Architecture
────────────
Single ArkSwinTransformer3D handles (B, 1, D, H, W) inputs by folding
the depth dimension into the batch before the 2D Swin encoder:

    (B, 1, D, H, W)
        ↓  rearrange → (B*D, 1, H, W)
        ↓  replicate → (B*D, 3, H, W)
    2D SwinTransformer.forward_features()
        ↓  pool → (B*D, F)
        ↓  reshape + mean → (B, F)          ← depth-averaged feature
    ArkProjector MLP  → (B, proj_dim)       ← consistency loss target
    TaskHead_i Linear → (B, n_classes_i)    ← classification logit

Binary datasets  → n_classes = 1  (single logit, BCEWithLogitsLoss)
Multi-class      → n_classes = N  (N logits,     CrossEntropyLoss)

Supported --model:
    swin_tiny   96-dim,  28M params  ← recommended for RTX 4060 + 28³ volumes
    swin_small  96-dim,  50M params
    swin_base   128-dim, 88M params
    swin_large  192-dim, 197M params
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from einops import rearrange

import timm.models.swin_transformer as swin
from timm.models.helpers import load_state_dict

from utils import remap_pretrained_keys_swin


# ─────────────────────────────────────────────────────────────────────────────
# Projection head  (3-layer MLP with BN — matches Ark+ Nature paper)
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
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Linear(h, o),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────

class ArkSwinTransformer3D(swin.SwinTransformer):
    """
    Unified Swin backbone for 3D volumetric inputs.

    Calling convention (matches original Ark+ trainer):
        feat, logit = model(x, head_n)
        feat → projected embedding  (B, proj_dim)   for consistency loss
        logit→ classification head  (B, n_classes)  for task loss
    """

    def __init__(self,
                 num_classes_list: list,
                 projector_features: int = None,
                 **swin_kwargs):
        super().__init__(**swin_kwargs)

        enc_dim  = self.num_features      # e.g. 768 for swin_tiny (96 × 2³)
        proj_dim = projector_features or enc_dim

        self.projector  = ArkProjector(enc_dim, enc_dim * 2, proj_dim)
        self.omni_heads = nn.ModuleList([
            nn.Linear(enc_dim, nc) for nc in num_classes_list
        ])

        self._enc_dim  = enc_dim
        self._proj_dim = proj_dim

        # Re-init classification heads (small std for stable early training)
        for head in self.omni_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0.0)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        """
        Collapse timm SwinTransformer output to (B, F).
        Handles all timm versions:
          (B, F)       — already pooled (timm ≥ 0.9 with global_pool='avg')
          (B, S, C)    — sequence  → mean over S
          (B, H, W, C) — spatial   → mean over H,W
        """
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 4:
            return x.mean(dim=[1, 2])
        raise ValueError(f"Unexpected feature shape from Swin: {x.shape}")

    @staticmethod
    def _to3ch(x: torch.Tensor) -> torch.Tensor:
        """Replicate single-channel input to 3 channels for patch embedding."""
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 2:
            return torch.cat([x, x[:, :1]], dim=1)
        return x   # already 3-channel

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract (B, enc_dim) feature vector from a 3D volume.
        x : (B, 1, D, H, W)
        """
        B, C, D, H, W = x.shape
        # Fold depth into batch
        x = rearrange(x, 'b c d h w -> (b d) c h w')   # (B*D, 1, H, W)
        x = self._to3ch(x)                               # (B*D, 3, H, W)
        feats = self._pool(super().forward_features(x))  # (B*D, enc_dim)
        # Average over depth slices → single feature per volume
        feats = feats.view(B, D, -1).mean(dim=1)         # (B, enc_dim)
        return feats

    # ── public API ────────────────────────────────────────────────────────────

    def forward(self,
                x: torch.Tensor,
                head_n: int = None):
        """
        Returns (proj_feat, logit) when head_n is given.
        proj_feat : (B, proj_dim)    — fed to consistency MSE loss
        logit     : (B, n_classes_i) — fed to task loss
        """
        enc  = self._encode(x)                    # (B, enc_dim)
        proj = self.projector(enc)                 # (B, proj_dim)

        if head_n is not None:
            return proj, self.omni_heads[head_n](enc)

        # Return all heads (not used during training)
        return [h(enc) for h in self.omni_heads]

    def generate_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected embeddings for downstream linear probing."""
        return self.projector(self._encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model_3d(args, num_classes_list: list) -> ArkSwinTransformer3D:
    """
    Build student or teacher model.

    args.model_name    : 'swin_tiny' | 'swin_small' | 'swin_base' | 'swin_large'
    args.crop_size     : image resolution (default 28 for MedMNIST 3D)
    num_classes_list   : [n_cls_dataset_0, n_cls_dataset_1, ...]
                         binary → 1,  multi-class N-way → N
    """
    name     = args.model_name
    img_size = getattr(args, 'crop_size', 28)
    pf       = getattr(args, 'projector_features', None)

    # patch_size=2 keeps 14×14 tokens at 28px; switch to 4 for larger inputs
    patch = 2 if img_size <= 64 else 4

    swin_kwargs = dict(
        img_size   = img_size,
        patch_size = patch,
        window_size= 7,
        num_classes= 0,          # disable timm's own head
    )

    if name == 'swin_tiny':
        model = ArkSwinTransformer3D(
            num_classes_list, pf,
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            **swin_kwargs)
    elif name == 'swin_small':
        model = ArkSwinTransformer3D(
            num_classes_list, pf,
            embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
            **swin_kwargs)
    elif name == 'swin_base':
        model = ArkSwinTransformer3D(
            num_classes_list, pf,
            embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            **swin_kwargs)
    elif name == 'swin_large':
        model = ArkSwinTransformer3D(
            num_classes_list, pf,
            embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            **swin_kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose: swin_tiny | swin_small | swin_base | swin_large"
        )

    # Optional pretrained weights
    pw = getattr(args, 'pretrained_weights', None)
    if pw:
        sd = (load_state_dict_from_url(pw, map_location='cpu')
              if pw.startswith('https') else load_state_dict(pw))
        for key in ('state_dict', 'model'):
            if key in sd:
                sd = sd[key]
                break
        # Drop keys that are always re-initialised
        for k in [k for k in sd if 'attn_mask' in k or 'omni_heads' in k]:
            del sd[k]
        msg = model.load_state_dict(sd, strict=False)
        print(f"[build_model_3d] Loaded pretrained weights: {msg}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[build_model_3d] {name}  |  enc_dim={model._enc_dim}  "
          f"|  params={n_params:.1f}M  |  heads={num_classes_list}")
    return model


def save_checkpoint(state: dict, filename: str = 'model'):
    torch.save(state, filename + '.pth.tar')
