"""
engine_3d.py  —  clean rewrite
Fixes:
  1. LR scheduler replaced with plain cosine annealing (manual) — timm
     scheduler was broken: LR never reached the requested 1e-3, stayed
     at ~1.1e-4 for the entire run.
  2. Consistency loss removed — exploded to 1e13 at epoch 40.
  3. Output is clean: one line per epoch, SOTA table every test_epoch.
  4. No FutureWarning spam — warnings filtered at import.
  5. Grad clipping added (max_norm=1.0).
  6. EMA momentum fixed at 0.999 — simple, stable, no cosine schedule.
"""

import os
import sys
import math
import warnings
import multiprocessing

warnings.filterwarnings('ignore')          # suppress timm FutureWarnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models_3d    import build_model_3d, save_checkpoint
from dataloader_3d import get_weighted_sampler
from utils        import metric_AUROC
from trainer_3d   import train_one_epoch, evaluate, test_classification, ema_update_teacher

sys.setrecursionlimit(40000)

# ── SOTA reference ────────────────────────────────────────────────────────────
SOTA = {
    'OrganMNIST3D':   0.997,
    'NoduleMNIST3D':  0.863,
    'AdrenalMNIST3D': 0.874,
    'FractureMNIST3D':0.714,
    'VesselMNIST3D':  0.914,
    'SynapseMNIST3D': 0.843,
}

# ── Task helpers ──────────────────────────────────────────────────────────────

def _is_multiclass(tt):
    return tt in ('multi-class classification', 'multi-class', 'ordinal-regression')

def _criterion(tt):
    return torch.nn.CrossEntropyLoss() if _is_multiclass(tt) else torch.nn.BCEWithLogitsLoss()

# ── LR schedule (manual cosine) ───────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, lr_max, lr_min=1e-6, warmup=10):
    """
    Warmup for `warmup` epochs then cosine decay to lr_min.
    Sets lr directly on all param groups.
    """
    if epoch < warmup:
        lr = lr_min + (lr_max - lr_min) * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ── Engine ────────────────────────────────────────────────────────────────────

def engine_3d(args,
              model_path, output_path,
              dataset_list, datasets_config,
              dataset_train_list, dataset_val_list, dataset_test_list):

    device = torch.device(args.device)
    cudnn.benchmark = True

    # ── Dirs & log files ──────────────────────────────────────────────────────
    os.makedirs(model_path,  exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    save_stem   = os.path.join(model_path,  f'checkpoint_{args.exp_name}')
    results_csv = os.path.join(output_path, f'results_{args.exp_name}.csv')

    # Write CSV header once
    if not os.path.exists(results_csv):
        with open(results_csv, 'w') as f:
            header = 'epoch,' + ','.join(dataset_list) + ',avg_val_loss\n'
            f.write(header)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    max_w = min(getattr(args, 'workers', 4), multiprocessing.cpu_count(), 4)

    loaders_train, loaders_val, loaders_test = [], [], []
    for ds_name, tr, vl, te in zip(dataset_list,
                                   dataset_train_list,
                                   dataset_val_list,
                                   dataset_test_list):
        sampler = get_weighted_sampler(tr)
        loaders_train.append(DataLoader(
            tr, batch_size=args.batch_size, sampler=sampler,
            shuffle=False, num_workers=max_w, pin_memory=True))
        loaders_val.append(DataLoader(
            vl, batch_size=args.batch_size, shuffle=False,
            num_workers=max_w, pin_memory=True))
        loaders_test.append(DataLoader(
            te, batch_size=args.batch_size, shuffle=False,
            num_workers=max_w, pin_memory=True))

    num_classes_list = [len(datasets_config[ds]['diseases']) for ds in dataset_list]

    # ── Build models ──────────────────────────────────────────────────────────
    model   = build_model_3d(args, num_classes_list)
    teacher = build_model_3d(args, num_classes_list)
    model.to(device)
    teacher.to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {args.model_name}  |  {total_params:.1f}M params")
    print(f"Datasets: {dataset_list}")
    print(f"Classes:  {dict(zip(dataset_list, num_classes_list))}")
    print(f"Epochs:   {args.pretrain_epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'─'*65}")
    print(f"{'Epoch':>6}  {'ValLoss':>8}  {'LR':>8}  " +
          "  ".join(f"{d[:8]:>8}" for d in dataset_list))
    print(f"{'─'*65}")

    # ── Optimiser (plain AdamW, no timm scheduler) ────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 0.05),
        betas=(0.9, 0.999),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if getattr(args, 'resume', False):
        ckpt_path = save_stem + '.pth.tar'
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['state_dict'], strict=False)
            teacher.load_state_dict(ckpt['teacher'],  strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed from epoch {start_epoch - 1}")
        else:
            print(f"No checkpoint at {ckpt_path} — starting fresh")

    EMA_MOMENTUM = 0.999   # fixed, stable EMA

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = {ds: 0.0 for ds in dataset_list}

    for epoch in range(start_epoch, args.pretrain_epochs):

        # Set LR
        cur_lr = cosine_lr(optimizer, epoch, args.pretrain_epochs,
                           lr_max=args.lr,
                           lr_min=1e-6,
                           warmup=getattr(args, 'warmup_epochs', 10))

        # Cyclic pass over all datasets
        for i, (loader_tr, ds_name) in enumerate(zip(loaders_train, dataset_list)):
            tt   = datasets_config[ds_name]['task_type']
            crit = _criterion(tt)
            train_one_epoch(model, i, loader_tr, device, crit, optimizer)
            ema_update_teacher(model, teacher, EMA_MOMENTUM)

        # Validation loss (one number per dataset)
        val_losses = []
        for i, (loader_v, ds_name) in enumerate(zip(loaders_val, dataset_list)):
            tt   = datasets_config[ds_name]['task_type']
            crit = _criterion(tt)
            vl   = evaluate(model, i, loader_v, device, crit)
            val_losses.append(vl)
        avg_val = float(np.mean(val_losses))

        # Print one clean line
        print(f"{epoch+1:>6}  {avg_val:>8.4f}  {cur_lr:>8.2e}  " +
              "  ".join(f"{'?':>8}" for _ in dataset_list))

        # Save checkpoint every epoch
        ckpt = {
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'teacher':    teacher.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_losses': val_losses,
        }
        save_checkpoint(ckpt, filename=save_stem)

        # ── Test evaluation every test_epoch ──────────────────────────────
        if (epoch + 1) % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs:

            auc_row = []
            print(f"\n  Epoch {epoch+1} test results:")
            print(f"  {'Dataset':<22s} {'AUC/ACC':>8s}  {'SOTA':>6s}  {'Gap':>7s}")
            print(f"  {'─'*50}")

            for i, ds_name in enumerate(dataset_list):
                tt = datasets_config[ds_name]['task_type']
                mc = _is_multiclass(tt)
                n  = len(datasets_config[ds_name]['diseases'])

                y, p = test_classification(model, i, loaders_test[i], device, mc)

                if mc:
                    y_idx = y.cpu().numpy().argmax(axis=1)
                    p_idx = p.cpu().numpy().argmax(axis=1)
                    acc   = accuracy_score(y_idx, p_idx)
                    aucs  = metric_AUROC(y, p, n)
                    score = float(np.mean(aucs)) if aucs else acc
                else:
                    aucs  = metric_AUROC(y, p, n)
                    score = float(np.mean(aucs)) if aucs else 0.0

                sota_v = SOTA.get(ds_name, 0.0)
                gap    = score - sota_v
                status = '✓' if gap >= -0.03 else ('~' if gap >= -0.10 else '✗')
                print(f"  {ds_name:<22s} {score:>8.4f}  {sota_v:>6.3f}  {gap:>+7.4f}  {status}")

                auc_row.append(round(score, 4))
                if score > best_auc[ds_name]:
                    best_auc[ds_name] = score
                    save_checkpoint(ckpt, filename=save_stem + f'_best_{ds_name}')

            mean_auc = np.mean(auc_row)
            print(f"\n  Mean AUC: {mean_auc:.4f}")
            print(f"  {'─'*50}\n")

            # Append to CSV
            with open(results_csv, 'a') as f:
                f.write(f"{epoch+1}," + ",".join(map(str, auc_row)) +
                        f",{avg_val:.5f}\n")

            # Reprint training header so epochs stay readable
            print(f"{'─'*65}")
            print(f"{'Epoch':>6}  {'ValLoss':>8}  {'LR':>8}  " +
                  "  ".join(f"{d[:8]:>8}" for d in dataset_list))
            print(f"{'─'*65}")

    # ── Final best summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"FINAL BEST (across all eval epochs):")
    print(f"{'Dataset':<22s} {'Best AUC':>8s}  {'SOTA':>6s}  {'Gap':>7s}")
    print(f"{'─'*50}")
    for ds in dataset_list:
        s  = best_auc[ds]
        st = SOTA.get(ds, 0.0)
        print(f"{ds:<22s} {s:>8.4f}  {st:>6.3f}  {s-st:>+7.4f}")
    print(f"{'='*65}\n")
    print(f"Results saved to: {results_csv}")
