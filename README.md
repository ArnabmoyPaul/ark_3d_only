# Ark+ 3D — Cyclic Pretraining on 6 × MedMNIST 3D Datasets

Focused replication of **Ark+** ([Ma et al., *Nature* 2025](https://doi.org/10.1038/s41586-025-09079-8)) applied to the 6 × MedMNIST 3D benchmark datasets.

---

## Why 3D-only?

Running all 18 MedMNIST datasets (12 2D + 6 3D) in one unified model causes the 2D classification tasks (11 datasets, large data) to dominate gradient updates. The 3D datasets (≤1,335 training samples each) learn nothing — the backbone optimises entirely for 2D features.

This repo trains on the 6 3D datasets **only**, giving the model full capacity to learn volumetric representations. This is the correct Ark+ principle applied to 3D: cyclic pretraining over heterogeneous 3D label spaces.

---

## SOTA targets

| Dataset | Task | #Train | SOTA |
|---------|------|--------|------|
| OrganMNIST3D | 11-class (organ) | 971 | 0.997 (ACC) |
| NoduleMNIST3D | binary (nodule) | 1,158 | 0.863 (AUC) |
| AdrenalMNIST3D | binary (adrenal) | 1,188 | 0.874 (AUC) |
| FractureMNIST3D | 3-class (fracture) | 1,027 | 0.714 (ACC) |
| VesselMNIST3D | binary (vessel) | 1,335 | 0.914 (AUC) |
| SynapseMNIST3D | binary (synapse) | 1,230 | 0.843 (AUC) |

Target: within **3%** of SOTA after 200 epochs.

---

## What was fixed vs run03

| # | Bug | Fix |
|---|-----|-----|
| 1 | `OrganMNIST3D` / `FractureMNIST3D` used `BCEWithLogitsLoss` (wrong loss for multi-class) | `_is_multiclass()` now matches both `'multi-class'` and `'multi-class classification'` |
| 2 | Binary datasets had 2-output one-hot BCE heads — confused gradients, AUC stuck at 0.50 | Config sets `diseases: [single_label]` → 1-logit head + correct BCE signal |
| 3 | `WeightedSampler` iterated entire 3D dataset on disk at startup (minutes) | Reads `.ds.labels` in-memory → instantaneous |
| 4 | `coff` could go negative when `momentum_teacher > 0.9` | Clamped to `[0, 0.5]` |
| 5 | Teacher not copied from student at init | `teacher.load_state_dict(model.state_dict())` before training |
| 6 | `metric_AUROC` called with wrong `nb_classes` for binary | Binary outputs `(N,1)` → `nb_classes=1` always correct |
| 7 | 18-dataset gradient interference | 3D-only run eliminates 2D dominance |

---

## Setup

```bash
conda create -n ark3d python=3.10
conda activate ark3d
pip install -r requirements.txt
```

---

## Usage

### Standard run (recommended)
```bash
python main_3d.py \
    --model swin_tiny \
    --pretrain_epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --momentum_teacher 0.9 \
    --warmup_epochs 10 \
    --test_epoch 5 \
    --exp_name run01 \
    --device cuda
```
Expected time on RTX 4060: **~3–4 hours**.

### Resume
```bash
python main_3d.py --exp_name run01 --resume \
    --model swin_tiny --pretrain_epochs 200 --batch_size 64 \
    --lr 1e-3 --momentum_teacher 0.9 --warmup_epochs 10
```

### Custom subset
```bash
python main_3d.py \
    --datasets OrganMNIST3D FractureMNIST3D NoduleMNIST3D \
    --model swin_tiny --pretrain_epochs 200 --batch_size 64 --exp_name run_3ds
```

### Higher resolution (64³ volumes)
```bash
python main_3d.py \
    --img_size 64 --batch_size 16 --model swin_small \
    --pretrain_epochs 200 --lr 5e-4 --exp_name run_64res
```

---

## Architecture

```
Input: (B, 1, D, H, W)
    ↓ fold depth → (B×D, 1, H, W)
    ↓ replicate  → (B×D, 3, H, W)
SwinTransformer (Tiny/Small/Base/Large)
    ↓ pool → (B×D, enc_dim)
    ↓ mean over D → (B, enc_dim)      ← depth-averaged volume feature
ArkProjector MLP (enc→2×enc→enc)
    ↓ (B, proj_dim)                   ← MSE consistency loss target
TaskHead_i   Linear (enc→n_classes_i)
    ↓ logits
```

**Loss per dataset:**
```
coff = clip((momentum[it] - 0.9) × 5,  0.0,  0.5)
loss = (1 - coff) × CrossEntropy_or_BCE  +  coff × MSE(proj_S, proj_T)
```

EMA teacher update at end of each dataset (epoch mode):
```
θ_T ← m × θ_T  +  (1 - m) × θ_S
m cosine-annealed from momentum_teacher → 1.0 over training
```

---

## Files

| File | Role |
|------|------|
| `main_3d.py` | Entry point, arg parsing, dataset setup |
| `engine_3d.py` | Cyclic training loop, EMA, checkpointing |
| `trainer_3d.py` | `train_one_epoch`, `evaluate`, `test_classification` |
| `models_3d.py` | `ArkSwinTransformer3D`, `ArkProjector`, model factory |
| `dataloader_3d.py` | `MedMNIST3DDataset`, augmentations, weighted sampler |
| `datasets_config_3d.yaml` | Per-dataset task type and class names |
| `utils.py` | Metrics, logging, cosine scheduler |

---

## Citation

```bibtex
@article{ma2025ark,
  title   = {A fully open AI foundation model applied to chest radiography},
  author  = {Ma, DongAo and Pang, Jiaxuan and Gotway, Michael B and Liang, Jianming},
  journal = {Nature},
  volume  = {643},
  pages   = {488--497},
  year    = {2025},
  doi     = {10.1038/s41586-025-09079-8}
}
```
