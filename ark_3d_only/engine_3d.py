"""
engine_3d.py
────────────
Ark+ cyclic pretraining engine — 6 × 3D MedMNIST datasets.

Cyclic training order (one full epoch):
  OrganMNIST3D → NoduleMNIST3D → AdrenalMNIST3D →
  FractureMNIST3D → VesselMNIST3D → SynapseMNIST3D

After each dataset the teacher is updated via EMA (epoch mode, default).
After every epoch the val loss is computed and used to step the LR scheduler.
Every `test_epoch` epochs the full test AUC / ACC is computed and logged.
"""

import os
import sys
import hashlib
import multiprocessing

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models_3d   import build_model_3d, save_checkpoint
from dataloader_3d import get_weighted_sampler
from utils       import metric_AUROC, cosine_scheduler
from trainer_3d  import train_one_epoch, evaluate, test_classification

from timm.scheduler import create_scheduler
from timm.optim     import create_optimizer

sys.setrecursionlimit(40000)


# ─────────────────────────────────────────────────────────────────────────────
# Task-type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_multiclass(task_type: str) -> bool:
    return task_type in ('multi-class classification', 'multi-class',
                         'ordinal-regression')

def _is_binary(task_type: str) -> bool:
    return task_type in ('binary classification', 'binary-class')

def _criterion(task_type: str):
    if _is_multiclass(task_type):
        return torch.nn.CrossEntropyLoss()
    return torch.nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

def engine_3d(args,
              model_path, output_path,
              dataset_list,
              datasets_config,
              dataset_train_list,
              dataset_val_list,
              dataset_test_list):
    """
    Main training entry point for 3D-only Ark+ cyclic pretraining.
    """

    device = torch.device(args.device)
    cudnn.benchmark = True

    # ── Logging ──────────────────────────────────────────────────────────────
    ds_hash   = hashlib.md5('_'.join(dataset_list).encode()).hexdigest()[:8]
    exp_short = f'Ark3D_{len(dataset_list)}ds_{ds_hash}'

    run_dir = os.path.join(model_path, exp_short, args.exp_name)
    os.makedirs(run_dir,     exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    log_file    = os.path.join(run_dir,     'train.log')
    output_file = os.path.join(output_path, f'{exp_short}_{args.exp_name}_results.txt')
    save_stem   = os.path.join(run_dir,     exp_short)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    max_w = min(args.workers, multiprocessing.cpu_count(), 4)
    print(f"DataLoader workers: {max_w}")

    # All 3D datasets use WeightedRandomSampler to balance classes
    loaders_train = []
    for ds_name, ds in zip(dataset_list, dataset_train_list):
        print(f"  [WeightedSampler] {ds_name}  (train={len(ds)})")
        sampler = get_weighted_sampler(ds)
        loaders_train.append(
            DataLoader(ds, batch_size=args.batch_size,
                       sampler=sampler, shuffle=False,
                       num_workers=max_w, pin_memory=True)
        )

    loaders_val = [
        DataLoader(d, batch_size=args.batch_size, shuffle=False,
                   num_workers=max_w, pin_memory=True)
        for d in dataset_val_list
    ]
    loaders_test = [
        DataLoader(d, batch_size=max(1, args.batch_size // 2),
                   shuffle=False, num_workers=max_w, pin_memory=True)
        for d in dataset_test_list
    ]

    num_classes_list = [
        len(datasets_config[ds]['diseases']) for ds in dataset_list
    ]
    print(f"num_classes_list: {dict(zip(dataset_list, num_classes_list))}")

    # ── Build student & teacher ───────────────────────────────────────────────
    model   = build_model_3d(args, num_classes_list)
    teacher = build_model_3d(args, num_classes_list)

    if torch.cuda.device_count() > 1:
        model   = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)

    model.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialise teacher = student (important — don't leave teacher random)
    teacher.load_state_dict(model.state_dict())

    # ── EMA momentum schedule ─────────────────────────────────────────────────
    n_ds = len(dataset_list)
    if args.ema_mode == 'epoch':
        # One EMA step per dataset per epoch → total steps = epochs × n_ds
        momentum_schedule = cosine_scheduler(
            args.momentum_teacher, 1.0,
            args.pretrain_epochs, n_ds,
        )
    else:
        total_iters = sum(len(dl) for dl in loaders_train)
        momentum_schedule = cosine_scheduler(
            args.momentum_teacher, 1.0,
            args.pretrain_epochs, total_iters,
        )

    # ── Optimiser & LR ────────────────────────────────────────────────────────
    optimizer        = create_optimizer(args, model)
    lr_scheduler, _  = create_scheduler(args, optimizer)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if getattr(args, 'resume', False):
        ckpt_path = save_stem + '.pth.tar'
        if os.path.isfile(ckpt_path):
            print(f"=> Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch'] + 1
            sd = ckpt['state_dict']
            if getattr(args, 'reinit_heads', False):
                sd = {k: v for k, v in sd.items()
                      if not k.startswith('omni_heads.')}
            model.load_state_dict(sd, strict=False)
            teacher.load_state_dict(ckpt['teacher'], strict=False)
            lr_scheduler.load_state_dict(ckpt['scheduler'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"=> Resumed from epoch {start_epoch - 1}")
        else:
            print(f"=> No checkpoint at {ckpt_path}")

    # ── Log setup ─────────────────────────────────────────────────────────────
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')
        f.write(f"Datasets: {dataset_list}\n")
        f.write(f"num_classes: {num_classes_list}\n\n")

    if args.mode != 'train':
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    test_results, test_results_teacher = [], []
    it = start_epoch * n_ds

    for epoch in range(start_epoch, args.pretrain_epochs):

        # ── Cyclic pass over all 6 datasets ───────────────────────────────
        for i, loader_tr in enumerate(loaders_train):
            task_type = datasets_config[dataset_list[i]]['task_type']
            crit      = _criterion(task_type)
            it = train_one_epoch(
                model, i, dataset_list[i],
                loader_tr, device, crit,
                optimizer, epoch,
                args.ema_mode, teacher, momentum_schedule, it,
            )

        # ── Validation ────────────────────────────────────────────────────
        val_losses = []
        for i, loader_v in enumerate(loaders_val):
            task_type = datasets_config[dataset_list[i]]['task_type']
            crit      = _criterion(task_type)
            vl = evaluate(model, i, loader_v, device, crit, dataset_list[i])
            val_losses.append(vl)

        avg_val = float(np.mean(val_losses))

        # LR schedule watch metric
        if args.val_loss_metric == 'average':
            watch = avg_val
        elif args.val_loss_metric in dataset_list:
            watch = val_losses[dataset_list.index(args.val_loss_metric)]
        else:
            watch = avg_val

        lr_scheduler.step(watch)
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:04d}  avg_val={avg_val:.5f}  lr={cur_lr:.2e}")

        # ── Save latest checkpoint ─────────────────────────────────────────
        ckpt = {
            'epoch':      epoch,
            'lossMIN':    val_losses,
            'state_dict': model.state_dict(),
            'teacher':    teacher.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  lr_scheduler.state_dict(),
            'args':       vars(args),
        }
        save_checkpoint(ckpt, filename=save_stem)

        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch:04d}  avg_val={avg_val:.5f}  lr={cur_lr:.2e}  "
                    f"val_losses={[round(v,5) for v in val_losses]}\n")

        # ── Periodic test evaluation ───────────────────────────────────────
        if epoch % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs:

            save_checkpoint(ckpt, filename=save_stem + f'_ep{epoch:04d}')

            t_res, t_res_teacher = [], []

            with open(output_file, 'a') as wf:
                wf.write(f"\n{'─'*60}\nEpoch {epoch:04d}:\n")

                for i, ds_name in enumerate(dataset_list):
                    task_type  = datasets_config[ds_name]['task_type']
                    diseases   = datasets_config[ds_name]['diseases']
                    mc         = _is_multiclass(task_type)
                    bi         = _is_binary(task_type)
                    n_cls      = len(diseases)

                    y_s, p_s = test_classification(
                        model,   i, loaders_test[i], device, mc, bi)
                    y_t, p_t = test_classification(
                        teacher, i, loaders_test[i], device, mc, bi)

                    if mc:
                        # Accuracy
                        y_idx = y_s.cpu().numpy().argmax(axis=1)
                        ps_idx= p_s.cpu().numpy().argmax(axis=1)
                        pt_idx= p_t.cpu().numpy().argmax(axis=1)
                        acc_s = accuracy_score(y_idx, ps_idx)
                        acc_t = accuracy_score(y_idx, pt_idx)
                        # Also compute mAUC
                        auc_s = metric_AUROC(y_s, p_s, n_cls)
                        auc_t = metric_AUROC(y_t, p_t, n_cls)
                        m_s   = float(np.mean(auc_s)) if auc_s else 0.0
                        m_t   = float(np.mean(auc_t)) if auc_t else 0.0
                        line  = (f"  {ds_name:<20s}  "
                                 f"S: ACC={acc_s:.4f} mAUC={m_s:.4f}  |  "
                                 f"T: ACC={acc_t:.4f} mAUC={m_t:.4f}\n")
                        t_res.append(m_s)
                        t_res_teacher.append(m_t)
                    else:
                        # AUROC (binary or multi-label)
                        auc_s = metric_AUROC(y_s, p_s, n_cls)
                        auc_t = metric_AUROC(y_t, p_t, n_cls)
                        m_s   = float(np.mean(auc_s)) if auc_s else 0.0
                        m_t   = float(np.mean(auc_t)) if auc_t else 0.0
                        line  = (f"  {ds_name:<20s}  "
                                 f"S: mAUC={m_s:.4f} {[round(v,4) for v in auc_s]}  |  "
                                 f"T: mAUC={m_t:.4f}\n")
                        t_res.append(m_s)
                        t_res_teacher.append(m_t)

                    print(line.strip())
                    wf.write(line)

            test_results.append(t_res)
            test_results_teacher.append(t_res_teacher)

            # Summary table
            print(f"\n{'Dataset':<20s} {'Student':>8s} {'Teacher':>8s}  {'SOTA':>8s}")
            sota = {'OrganMNIST3D':0.997,'NoduleMNIST3D':0.863,'AdrenalMNIST3D':0.874,
                    'FractureMNIST3D':0.714,'VesselMNIST3D':0.914,'SynapseMNIST3D':0.843}
            for ds, s, t in zip(dataset_list, t_res, t_res_teacher):
                st = sota.get(ds, 0.0)
                gap = s - st
                flag_str = '✓' if gap >= -0.03 else '✗'
                print(f"{ds:<20s} {s:>8.4f} {t:>8.4f}  {st:>8.4f}  {flag_str} ({gap:+.4f})")
            print()

    # ── Final summary ─────────────────────────────────────────────────────────
    with open(output_file, 'a') as wf:
        wf.write(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}\n")
        wf.write(f"{'Dataset':<20s} {'Student':>8s} {'Teacher':>8s} {'SOTA':>8s}\n")
        sota = {'OrganMNIST3D':0.997,'NoduleMNIST3D':0.863,'AdrenalMNIST3D':0.874,
                'FractureMNIST3D':0.714,'VesselMNIST3D':0.914,'SynapseMNIST3D':0.843}
        if test_results:
            final = test_results[-1]
            final_t = test_results_teacher[-1]
            for ds, s, t in zip(dataset_list, final, final_t):
                wf.write(f"{ds:<20s} {s:>8.4f} {t:>8.4f} {sota.get(ds,0):>8.4f}\n")

    print("\nTraining complete.")
