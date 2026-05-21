"""
trainer_3d.py
─────────────
Training primitives for Ark+ 3D-only cyclic pretraining.

train_one_epoch   — one full dataset pass, returns updated `it`
ema_update_teacher— EMA weight copy from student to teacher
evaluate          — validation loss
test_classification — inference + collect (y_true, y_pred) tensors
"""

import time
import numpy as np
import torch
from tqdm import tqdm

from utils import MetricLogger, ProgressLogger

# ── optional wandb ────────────────────────────────────────────────────────────
try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False

def _wlog(k, v):
    if _WANDB:
        try:
            _wandb.log({k: v})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

def ema_update_teacher(model, teacher, momentum_schedule, it):
    """Copy student weights into teacher via exponential moving average."""
    with torch.no_grad():
        m = momentum_schedule[it]
        for p_s, p_t in zip(model.parameters(), teacher.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, head_n, ds_name,
                    loader, device, criterion,
                    optimizer, epoch,
                    ema_mode, teacher, momentum_schedule, it):
    """
    One full pass over `loader`.

    Loss = (1 - coff) × task_loss  +  coff × MSE(proj_student, proj_teacher)

    coff ramps from 0 (pure task loss at start) toward 0.5 as momentum
    approaches 1.0.  Clamped to [0, 0.5] — prevents negative weights when
    momentum_teacher is initialised above 0.9.

    Returns
    ───────
    it : updated global iteration counter  (MUST be captured by caller)
    """
    t_batch  = MetricLogger('Time',                   ':6.3f')
    l_task   = MetricLogger(f'{ds_name}_task',        ':.4e')
    l_consist= MetricLogger(f'{ds_name}_consist',     ':.4e')
    progress = ProgressLogger(
        len(loader), [t_batch, l_task, l_consist],
        prefix=f'Epoch [{epoch}] {ds_name}',
    )

    model.train()
    MSE  = torch.nn.MSELoss()
    coff = float(np.clip((momentum_schedule[it] - 0.9) * 5, 0.0, 0.5))
    end  = time.time()

    for i, (s1, s2, targets) in enumerate(loader):
        s1      = s1.float().to(device)          # (B,1,D,H,W)
        s2      = s2.float().to(device)
        targets = targets.to(device)             # long or float(1,)

        feat_t, pred_t = teacher(s2, head_n)
        feat_s, pred_s = model(s1,   head_n)

        loss_task    = criterion(pred_s, targets)
        loss_consist = MSE(feat_s, feat_t)
        loss         = (1 - coff) * loss_task + coff * loss_consist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l_task.update(loss_task.item(),    s1.size(0))
        l_consist.update(loss_consist.item(), s1.size(0))
        t_batch.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

        if ema_mode == 'iteration':
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == 'epoch':
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    _wlog(f'train_task_{ds_name}',    l_task.avg)
    _wlog(f'train_consist_{ds_name}', l_consist.avg)

    return it   # ← caller must store this return value


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, head_n, loader, device, criterion, ds_name):
    """Returns average validation loss over the loader."""
    model.eval()
    losses   = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
        len(loader), [losses], prefix=f'Val {ds_name}: ')

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(loader):
            samples = samples.float().to(device)
            targets = targets.to(device)
            _, out  = model(samples, head_n)
            loss    = criterion(out, targets)
            losses.update(loss.item(), samples.size(0))
            if i % 50 == 0:
                progress.display(i)

    return losses.avg


# ─────────────────────────────────────────────────────────────────────────────
# Test / inference
# ─────────────────────────────────────────────────────────────────────────────

def test_classification(model, head_n, loader,
                        device, multiclass=False, binary=False):
    """
    Collect ground-truth and prediction tensors over the full test set.

    Parameters
    ──────────
    multiclass : True  → softmax → returns (B, n_cls) probs + (B, n_cls) one-hot GT
    binary     : True  → sigmoid on single logit (B,1) → returns (B,1) prob + GT
    (else)     : multi-label sigmoid

    Returns
    ───────
    y_test : FloatTensor on device  — ground truth in AUC-compatible format
    p_test : FloatTensor on device  — model probabilities
    """
    model.eval()
    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for samples, _, targets in tqdm(loader, desc=f'Test head={head_n}'):
            samples = samples.float().to(device)
            targets = targets.to(device)

            _, out = model(samples, head_n)    # out: (B, n_cls) or (B,1)

            # Activate
            if multiclass:
                probs = torch.softmax(out, dim=1)
            else:
                probs = torch.sigmoid(out)     # binary (B,1) or multi-label

            # Format ground truth to match probs shape for AUROC
            if multiclass:
                B, n_cls = probs.shape
                y_oh = torch.zeros(B, n_cls, device=device)
                y_oh.scatter_(1, targets.view(-1, 1).long(), 1.0)
                y = y_oh
            else:
                y = targets.float()            # already (B,1) or (B,n_cls)

            y_test = torch.cat([y_test, y],     dim=0)
            p_test = torch.cat([p_test, probs], dim=0)

    return y_test, p_test
