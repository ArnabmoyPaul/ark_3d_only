"""
dataloader_3d.py
────────────────
MedMNIST 3D Dataset classes for Ark+ cyclic pretraining.
Covers all 6 × 3D MedMNIST datasets.

Each dataset returns:  (student_view, teacher_view, label)
  student_view : torch float32  (1, D, H, W)  — strong augmentation
  teacher_view : torch float32  (1, D, H, W)  — weak/clean augmentation
  label        :
    binary classification  → torch float32  shape (1,)   [0. or 1.]
    multi-class            → torch long     shape ()      [class index]

Student augmentation — Models Genesis (Zhou et al., MIA 2022) + standard:
  1. Non-linear intensity  — Bézier curve transforms intensity distribution
                             forces model to learn organ appearance
  2. Local pixel shuffling — corrupts local texture inside small windows
                             forces model to learn boundaries and textures
  3. Inner-cutout          — masks inner region, forces local interpolation
  4. Outer-cutout          — masks outer region, forces spatial extrapolation
  5. Axis flips + 90° rot  — standard geometric augmentation
  6. Gaussian noise        — standard noise augmentation
Each transform applied independently with p=0.5 (outer/inner mutually exclusive).
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import medmnist
from medmnist import INFO

# Registry
MEDMNIST_3D_FLAGS = [
    'organmnist3d',
    'nodulemnist3d',
    'adrenalmnist3d',
    'fracturemnist3d',
    'vesselmnist3d',
    'synapsemnist3d',
]


# ─────────────────────────────────────────────────────────────────────────────
# Models Genesis transforms (3D versions)
# ─────────────────────────────────────────────────────────────────────────────

def _bezier_nonlinear(vol: np.ndarray) -> np.ndarray:
    """
    Non-linear intensity transformation via monotonic Bézier curve.
    Forces model to learn organ appearance from intensity distributions.
    vol: float32 (D,H,W) in [0,1]  →  returns same shape in [0,1]
    """
    # Two control points randomly chosen — curve stays monotonic
    # by constraining P1.x < P2.x and both between 0 and 1
    p1x = random.uniform(0.1, 0.4)
    p1y = random.uniform(0.0, 0.6) if random.random() < 0.5 else random.uniform(0.4, 1.0)
    p2x = random.uniform(0.6, 0.9)
    p2y = random.uniform(0.4, 1.0) if random.random() < 0.5 else random.uniform(0.0, 0.6)

    # Evaluate cubic Bézier B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    # P0=(0,0), P3=(1,1) so curve maps [0,1]→[0,1]
    t = vol.flatten()
    # Approximate: use lookup table over 256 points
    t_vals = np.linspace(0.0, 1.0, 256)
    y_vals = (3*(1-t_vals)**2*t_vals*p1y +
              3*(1-t_vals)*t_vals**2*p2y +
              t_vals**3)
    y_vals = np.clip(y_vals, 0.0, 1.0)
    # Map each voxel intensity through the curve using interpolation
    out = np.interp(vol, t_vals, y_vals)
    return out.astype(np.float32)


def _local_pixel_shuffling(vol: np.ndarray, n_windows: int = 500) -> np.ndarray:
    """
    Local pixel shuffling: randomly shuffle voxels inside small windows.
    Corrupts local texture/boundary while preserving global structure.
    Forces model to learn local boundaries and texture patterns.
    vol: float32 (D,H,W) in [0,1]
    """
    out = vol.copy()
    D, H, W = vol.shape

    for _ in range(n_windows):
        # Random window size — small enough to keep global structure
        wd = random.randint(2, max(2, D // 5))
        wh = random.randint(2, max(2, H // 5))
        ww = random.randint(2, max(2, W // 5))

        # Random anchor — max(0,...) prevents randint(0,0) crash
        d0 = random.randint(0, max(0, D - wd))
        h0 = random.randint(0, max(0, H - wh))
        w0 = random.randint(0, max(0, W - ww))

        region = out[d0:d0+wd, h0:h0+wh, w0:w0+ww]
        shape  = region.shape            # actual clipped shape
        patch  = region.flatten().copy()
        np.random.shuffle(patch)
        out[d0:d0+wd, h0:h0+wh, w0:w0+ww] = patch.reshape(shape)

    return out


def _inner_cutout(vol: np.ndarray, max_ratio: float = 0.25) -> np.ndarray:
    """
    Mask a random inner region with zero (black).
    Forces model to learn local continuity via interpolation.
    Cutout region ≤ 25% of total volume.
    """
    out = vol.copy()
    D, H, W = vol.shape

    wd = random.randint(max(1, D // 10), max(1, int(D * max_ratio)))
    wh = random.randint(max(1, H // 10), max(1, int(H * max_ratio)))
    ww = random.randint(max(1, W // 10), max(1, int(W * max_ratio)))

    # max(0, ...) prevents randint(0, 0) crash when window == full dimension
    d0 = random.randint(0, max(0, D - wd))
    h0 = random.randint(0, max(0, H - wh))
    w0 = random.randint(0, max(0, W - ww))

    out[d0:d0+wd, h0:h0+wh, w0:w0+ww] = 0.0
    return out


def _outer_cutout(vol: np.ndarray, max_ratio: float = 0.25) -> np.ndarray:
    """
    Keep a random inner region exposed, mask everything outside with zero.
    Forces model to learn global geometry and spatial layout via extrapolation.
    Window covers 50–90% of each dimension (never the full axis to keep anchor valid).
    """
    out = np.zeros_like(vol)
    D, H, W = vol.shape

    wd = random.randint(max(1, int(D * 0.5)), max(1, int(D * 0.9)))
    wh = random.randint(max(1, int(H * 0.5)), max(1, int(H * 0.9)))
    ww = random.randint(max(1, int(W * 0.5)), max(1, int(W * 0.9)))

    # max(0, ...) prevents randint(0, 0) crash
    d0 = random.randint(0, max(0, D - wd))
    h0 = random.randint(0, max(0, H - wh))
    w0 = random.randint(0, max(0, W - ww))

    out[d0:d0+wd, h0:h0+wh, w0:w0+ww] = vol[d0:d0+wd, h0:h0+wh, w0:w0+ww]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Student / Teacher / Val augmentation functions
# ─────────────────────────────────────────────────────────────────────────────

def aug_student(vol: np.ndarray) -> torch.Tensor:
    """
    Strong augmentation for the student view.
    vol : float32 numpy (D, H, W) in [0, 1]
    Returns torch float32 (1, D, H, W) normalised to [-1, 1]

    Reduced intensity vs original — with only 16 batches/epoch,
    overly aggressive corruption prevents stable representation learning.
    Models Genesis transforms kept but at lower probability.
    """
    # ── Geometric ─────────────────────────────────────────────────────────
    for ax in range(3):
        if random.random() < 0.5:
            vol = np.flip(vol, axis=ax).copy()

    if random.random() < 0.5:
        k   = random.randint(1, 3)
        vol = np.rot90(vol, k=k, axes=(1, 2)).copy()

    # ── Models Genesis Transform 1: Non-linear intensity (p=0.3) ──────────
    if random.random() < 0.3:
        vol = _bezier_nonlinear(vol)

    # ── Models Genesis Transform 2: Local pixel shuffling (p=0.3) ─────────
    if random.random() < 0.3:
        vol = _local_pixel_shuffling(vol, n_windows=150)

    # ── Models Genesis Transforms 3 & 4: Cutout (p=0.2 each) ─────────────
    cutout_choice = random.random()
    if cutout_choice < 0.15:
        vol = _inner_cutout(vol)
    elif cutout_choice < 0.30:
        vol = _outer_cutout(vol)

    # ── Gaussian noise (p=0.2) ────────────────────────────────────────────
    if random.random() < 0.2:
        noise = np.random.normal(0, 0.01, vol.shape).astype(np.float32)
        vol   = np.clip(vol + noise, 0.0, 1.0)

    # ── Normalise to [-1, 1] ──────────────────────────────────────────────
    vol = (vol.astype(np.float32) - 0.5) / 0.5
    return torch.tensor(vol[np.newaxis], dtype=torch.float32)


def aug_teacher(vol: np.ndarray) -> torch.Tensor:
    """
    Weak augmentation for the teacher view.
    Only mild depth flip + tiny intensity scale — keeps teacher signal stable.
    """
    if random.random() < 0.3:
        vol = np.flip(vol, axis=0).copy()   # depth flip only
    if random.random() < 0.3:
        alpha = random.uniform(0.9, 1.1)
        vol   = np.clip(vol * alpha, 0.0, 1.0).astype(np.float32)
    vol = (vol - 0.5) / 0.5
    return torch.tensor(vol[np.newaxis], dtype=torch.float32)


def aug_val(vol: np.ndarray) -> torch.Tensor:
    """No augmentation — deterministic normalisation only."""
    vol = (vol.astype(np.float32) - 0.5) / 0.5
    return torch.tensor(vol[np.newaxis], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MedMNIST3DDataset(Dataset):
    """
    Wraps any MedMNIST 3D flag.

    Parameters
    ----------
    flag     : medmnist flag string, e.g. 'organmnist3d'
    split    : 'train' | 'val' | 'test'
    mode     : 'train' → dual augmented views
               'val'/'test' → single deterministic view
    download : auto-download if not cached
    root     : cache dir (default ~/.medmnist)

    Label format
    ────────────
    binary-class  → torch.float32  shape (1,)   for BCEWithLogitsLoss
    multi-class   → torch.long     shape ()      for CrossEntropyLoss
    """

    def __init__(self, flag: str, split: str = 'train',
                 mode: str = 'train',
                 download: bool = True,
                 root: str = None):

        assert flag in MEDMNIST_3D_FLAGS, \
            f"Unknown 3D flag '{flag}'. Valid: {MEDMNIST_3D_FLAGS}"

        info      = INFO[flag]
        DataClass = getattr(medmnist, info['python_class'])

        kw = dict(split=split, download=download)
        if root is not None:
            kw['root'] = root

        self.ds        = DataClass(**kw)
        self.mode      = mode
        self.task      = info['task']           # 'binary-class' | 'multi-class'
        self.n_classes = len(info['label'])
        self.flag      = flag

    # ── label helpers ─────────────────────────────────────────────────────────

    def _make_label(self, raw) -> torch.Tensor:
        """
        binary-class : float scalar wrapped in shape (1,)
        multi-class  : long scalar
        """
        arr = np.array(raw).squeeze()
        idx = int(arr) if arr.ndim == 0 else int(arr[0])

        if self.task == 'binary-class':
            return torch.tensor([float(idx)], dtype=torch.float32)
        else:
            return torch.tensor(idx, dtype=torch.long)

    # ── data loading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img_raw, lbl_raw = self.ds[idx]

        # medmnist 3D returns numpy uint8 (D, H, W) or (D, H, W, 1)
        arr = np.array(img_raw).astype(np.float32) / 255.0
        if arr.ndim == 4:
            arr = arr[..., 0]       # (D,H,W,1) → (D,H,W)

        label = self._make_label(lbl_raw)

        if self.mode == 'train':
            student = aug_student(arr)
            teacher = aug_teacher(arr)
        else:
            v       = aug_val(arr)
            student = v
            teacher = v

        return student, teacher, label


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler  (instantaneous — reads in-memory labels)
# ─────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(dataset: MedMNIST3DDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that balances classes each epoch.

    Reads the pre-loaded label array directly from the underlying medmnist
    Dataset object — no per-sample __getitem__ call needed.

    Works for both binary (0/1) and multi-class (0…N-1) labels.
    """
    raw    = np.array(dataset.ds.labels).squeeze()
    if raw.ndim == 2:
        raw = raw[:, 0]
    labels = raw.astype(int)

    classes, counts = np.unique(labels, return_counts=True)
    class_weight    = 1.0 / counts.astype(np.float64)
    sample_weights  = torch.tensor(
        [class_weight[np.searchsorted(classes, l)] for l in labels],
        dtype=torch.float64,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_datasets(flag: str,
                      download: bool = True,
                      root: str = None):
    """
    Returns (train_ds, val_ds, test_ds) for the given 3D MedMNIST flag.
    """
    tr = MedMNIST3DDataset(flag, 'train', 'train', download, root)
    vl = MedMNIST3DDataset(flag, 'val',   'val',   download, root)
    te = MedMNIST3DDataset(flag, 'test',  'test',  download, root)
    return tr, vl, te
