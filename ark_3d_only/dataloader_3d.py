"""
dataloader_3d.py
────────────────
MedMNIST 3D Dataset classes for Ark+ cyclic pretraining.
Covers all 6 × 3D MedMNIST datasets.

Each dataset returns:  (student_view, teacher_view, label)
  student_view : torch float32  (1, D, H, W)  — strong augmentation
  teacher_view : torch float32  (1, D, H, W)  — weak augmentation
  label        :
    binary classification  → torch float32  shape (1,)   [0. or 1.]
    multi-class            → torch long     shape ()      [class index]

Key design choices
──────────────────
1. Single-logit heads for binary tasks (BCEWithLogitsLoss).
   Previous run03 bug: 2-output one-hot BCELoss — confused gradients,
   AUC stuck near 0.50.
2. WeightedRandomSampler reads .labels from memory — no disk loop.
3. Augmentation separates student (strong) from teacher (weak/clean)
   exactly as in the original Ark+ CXR dataloader.
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
# 3D Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def aug_student(vol: np.ndarray) -> torch.Tensor:
    """
    Strong augmentation for the student view.
    vol : float32 numpy (D, H, W) in [0, 1]
    Returns torch float32 (1, D, H, W) normalised to [-1, 1]
    """
    # Random axis flips
    for ax in range(3):
        if random.random() < 0.5:
            vol = np.flip(vol, axis=ax).copy()

    # Random 90° in-plane rotation (axial)
    if random.random() < 0.5:
        k = random.randint(1, 3)
        vol = np.rot90(vol, k=k, axes=(1, 2)).copy()

    # Random intensity shift + scale
    alpha = random.uniform(0.75, 1.25)
    beta  = random.uniform(-0.15, 0.15)
    vol   = np.clip(vol * alpha + beta, 0.0, 1.0).astype(np.float32)

    # Random Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, vol.shape).astype(np.float32)
        vol   = np.clip(vol + noise, 0.0, 1.0)

    # Normalise to [-1, 1]
    vol = (vol - 0.5) / 0.5
    return torch.tensor(vol[np.newaxis], dtype=torch.float32)   # (1,D,H,W)


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
