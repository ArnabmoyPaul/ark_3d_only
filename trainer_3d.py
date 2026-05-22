"""
trainer_3d.py  —  clean rewrite
Fixes:
  1. Consistency loss DISABLED — was exploding to 1e13 at epoch 40
     (MSE between projector outputs diverges when EMA momentum → 1)
     Pure task loss only until classification works.
  2. No tqdm / progress spam — silent training, one line per epoch printed
     by engine_3d.
  3. test_classification: no tqdm output, just returns tensors.
"""

import torch
from tqdm import tqdm


# ── EMA ──────────────────────────────────────────────────────────────────────

def ema_update_teacher(model, teacher, momentum):
    with torch.no_grad():
        for p_s, p_t in zip(model.parameters(), teacher.parameters()):
            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.detach().data)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, head_n, loader, device, criterion, optimizer):
    """
    One full pass. Pure task loss only — no consistency term.
    Returns average task loss for this dataset.
    """
    model.train()
    total_loss = 0.0
    total_n    = 0

    for s1, s2, targets in loader:
        s1      = s1.float().to(device)
        targets = targets.to(device)

        _, pred = model(s1, head_n)
        loss    = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * s1.size(0)
        total_n    += s1.size(0)

    return total_loss / total_n


# ── Validation ────────────────────────────────────────────────────────────────

def evaluate(model, head_n, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_n    = 0
    with torch.no_grad():
        for samples, _, targets in loader:
            samples = samples.float().to(device)
            targets = targets.to(device)
            _, out  = model(samples, head_n)
            loss    = criterion(out, targets)
            total_loss += loss.item() * samples.size(0)
            total_n    += samples.size(0)
    return total_loss / total_n


# ── Test / inference ──────────────────────────────────────────────────────────

def test_classification(model, head_n, loader, device,
                        multiclass=False):
    """
    Returns (y_true, y_pred) as FloatTensors.
    multiclass=True  → softmax + one-hot GT
    else             → sigmoid (binary single logit)
    """
    model.eval()
    y_all = torch.FloatTensor().to(device)
    p_all = torch.FloatTensor().to(device)

    with torch.no_grad():
        for samples, _, targets in loader:
            samples = samples.float().to(device)
            targets = targets.to(device)

            _, out = model(samples, head_n)

            if multiclass:
                probs = torch.softmax(out, dim=1)
                B, n_cls = probs.shape
                y_oh = torch.zeros(B, n_cls, device=device)
                y_oh.scatter_(1, targets.view(-1, 1).long(), 1.0)
                y = y_oh
            else:
                probs = torch.sigmoid(out)
                y     = targets.float()

            y_all = torch.cat([y_all, y],     dim=0)
            p_all = torch.cat([p_all, probs], dim=0)

    return y_all, p_all
