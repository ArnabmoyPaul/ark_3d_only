"""
engine_3d.py  —  parallel multi-task training (not cyclic)

Why cyclic pretraining failed for MedMNIST 3D:
  - Only 16 batches/epoch per dataset (1000 samples / 64 batch)
  - Cyclic: train Organ → Nodule → ... → Synapse → repeat
  - Each dataset partially overwrites what was learned on previous ones
  - With 704K samples (Ark+) this is fine. With 1000 it's catastrophic forgetting.

New approach: all 6 datasets in a SINGLE epoch
  - One DataLoader per dataset
  - Each batch: sample one dataset round-robin, do one gradient step
  - All tasks share the encoder, each has its own head
  - ~96 gradient steps per epoch (16 per dataset × 6 datasets)
  - No forgetting — tasks update the shared encoder simultaneously
"""

import os
import sys
import math
import warnings
import itertools
import multiprocessing

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models_3d    import build_model_3d, save_checkpoint
from dataloader_3d import get_weighted_sampler
from utils        import metric_AUROC
from trainer_3d   import evaluate, test_classification, ema_update_teacher

sys.setrecursionlimit(40000)

SOTA = {
    'OrganMNIST3D':   0.997,
    'NoduleMNIST3D':  0.863,
    'AdrenalMNIST3D': 0.874,
    'FractureMNIST3D':0.714,
    'VesselMNIST3D':  0.914,
    'SynapseMNIST3D': 0.843,
}

def _is_multiclass(tt):
    return tt in ('multi-class classification', 'multi-class', 'ordinal-regression')

def _criterion(tt):
    return torch.nn.CrossEntropyLoss() if _is_multiclass(tt) else torch.nn.BCEWithLogitsLoss()

def cosine_lr(optimizer, epoch, total_epochs, lr_max, lr_min=1e-6, warmup=10):
    if epoch < warmup:
        lr = lr_min + (lr_max - lr_min) * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch_multitask(model, loaders, criterions, optimizer, device):
    """
    Round-robin over all datasets in one epoch.
    Each iteration: pick next dataset, take one batch, update.
    Returns avg loss across all batches.
    """
    model.train()
    total_loss = 0.0
    total_n    = 0

    # Create iterators — will cycle through all datasets together
    iters = [iter(ld) for ld in loaders]
    n_datasets = len(loaders)
    n_batches  = max(len(ld) for ld in loaders)

    for batch_idx in range(n_batches):
        for ds_idx in range(n_datasets):
            try:
                s1, s2, targets = next(iters[ds_idx])
            except StopIteration:
                iters[ds_idx] = iter(loaders[ds_idx])
                s1, s2, targets = next(iters[ds_idx])

            s1      = s1.float().to(device)
            targets = targets.to(device)

            _, pred = model(s1, ds_idx)
            loss    = criterions[ds_idx](pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * s1.size(0)
            total_n    += s1.size(0)

    ema_update_teacher(model, model, 1.0)  # dummy — teacher not used
    return total_loss / max(total_n, 1)


def engine_3d(args,
              model_path, output_path,
              dataset_list, datasets_config,
              dataset_train_list, dataset_val_list, dataset_test_list):

    device = torch.device(args.device)
    cudnn.benchmark = True

    os.makedirs(model_path,  exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    save_stem   = os.path.join(model_path,  f'checkpoint_{args.exp_name}')
    results_csv = os.path.join(output_path, f'results_{args.exp_name}.csv')

    if not os.path.exists(results_csv):
        with open(results_csv, 'w') as f:
            f.write('epoch,' + ','.join(dataset_list) + ',avg_val_loss\n')

    # Windows multiprocessing spawns new Python processes which re-import
    # scipy/sklearn — this can exhaust the pagefile and crash with
    # "DLL load failed: paging file too small".
    # Safe fix: use 0 workers (main-process loading) on Windows.
    import platform
    max_w = 0 if platform.system() == 'Windows' else min(
        getattr(args, 'workers', 4), multiprocessing.cpu_count(), 4)

    loaders_train, loaders_val, loaders_test = [], [], []
    for tr, vl, te in zip(dataset_train_list, dataset_val_list, dataset_test_list):
        sampler = get_weighted_sampler(tr)
        loaders_train.append(DataLoader(tr, batch_size=args.batch_size,
                                         sampler=sampler, shuffle=False,
                                         num_workers=max_w, pin_memory=True))
        loaders_val.append(DataLoader(vl, batch_size=args.batch_size,
                                       shuffle=False, num_workers=max_w, pin_memory=True))
        loaders_test.append(DataLoader(te, batch_size=args.batch_size,
                                        shuffle=False, num_workers=max_w, pin_memory=True))

    criterions       = [_criterion(datasets_config[ds]['task_type']) for ds in dataset_list]
    num_classes_list = [len(datasets_config[ds]['diseases']) for ds in dataset_list]

    model = build_model_3d(args, num_classes_list)
    # No separate teacher — single model, no EMA needed for pure supervised training
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {getattr(args,'model_name','3D-ResNet')}  |  {total_params:.1f}M params")
    print(f"Datasets: {dataset_list}")
    print(f"Classes:  {dict(zip(dataset_list, num_classes_list))}")
    print(f"Epochs:   {args.pretrain_epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'─'*65}")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  {'LR':>8}")
    print(f"{'─'*65}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=getattr(args, 'weight_decay', 0.05),
                                   betas=(0.9, 0.999))

    start_epoch = 0
    if getattr(args, 'resume', False):
        ckpt_path = save_stem + '.pth.tar'
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed from epoch {start_epoch - 1}")

    best_auc = {ds: 0.0 for ds in dataset_list}

    for epoch in range(start_epoch, args.pretrain_epochs):

        cur_lr = cosine_lr(optimizer, epoch, args.pretrain_epochs,
                           lr_max=args.lr, lr_min=1e-6,
                           warmup=getattr(args, 'warmup_epochs', 10))

        # ── Multi-task training (all datasets in one epoch) ───────────────
        train_loss = train_one_epoch_multitask(
            model, loaders_train, criterions, optimizer, device)

        # ── Validation loss ───────────────────────────────────────────────
        val_losses = [evaluate(model, i, loaders_val[i], device, criterions[i])
                      for i, ds in enumerate(dataset_list)]
        avg_val = float(np.mean(val_losses))

        print(f"{epoch+1:>6}  {train_loss:>10.4f}  {avg_val:>8.4f}  {cur_lr:>8.2e}")

        ckpt = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'val_losses': val_losses}
        save_checkpoint(ckpt, filename=save_stem)

        # ── Test every test_epoch ─────────────────────────────────────────
        if (epoch + 1) % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs:

            auc_row = []
            print(f"\n  Epoch {epoch+1} test results:")
            print(f"  {'Dataset':<22s} {'AUC/ACC':>8s}  {'SOTA':>6s}  {'Gap':>7s}")
            print(f"  {'─'*52}")

            for i, ds_name in enumerate(dataset_list):
                mc = _is_multiclass(datasets_config[ds_name]['task_type'])
                n  = len(datasets_config[ds_name]['diseases'])
                y, p = test_classification(model, i, loaders_test[i], device, mc)

                if mc:
                    y_idx = y.cpu().numpy().argmax(axis=1)
                    p_idx = p.cpu().numpy().argmax(axis=1)
                    aucs  = metric_AUROC(y, p, n)
                    score = float(np.mean(aucs)) if aucs else accuracy_score(y_idx, p_idx)
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
            print(f"  {'─'*52}\n")

            with open(results_csv, 'a') as f:
                f.write(f"{epoch+1}," + ",".join(map(str, auc_row)) + f",{avg_val:.5f}\n")

            print(f"{'─'*65}")
            print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  {'LR':>8}")
            print(f"{'─'*65}")

    print(f"\n{'='*65}")
    print("FINAL BEST:")
    print(f"{'Dataset':<22s} {'Best':>8s}  {'SOTA':>6s}  {'Gap':>7s}")
    print(f"{'─'*52}")
    for ds in dataset_list:
        s = best_auc[ds]; st = SOTA.get(ds, 0.0)
        print(f"{ds:<22s} {s:>8.4f}  {st:>6.3f}  {s-st:>+7.4f}")
    print(f"{'='*65}\n")
    print(f"Results saved to: {results_csv}")
