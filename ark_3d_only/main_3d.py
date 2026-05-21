"""
main_3d.py
──────────
Entry point for Ark+ cyclic pretraining on 6 × MedMNIST 3D datasets.

Quick-start
───────────
# Standard 200-epoch run (recommended for near-SOTA):
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

# Resume interrupted run:
python main_3d.py --exp_name run01 --resume \
    --model swin_tiny --pretrain_epochs 200 --batch_size 64 \
    --lr 1e-3 --momentum_teacher 0.9 --warmup_epochs 10

# Custom subset of 3D datasets:
python main_3d.py \
    --datasets OrganMNIST3D NoduleMNIST3D FractureMNIST3D \
    --model swin_tiny --pretrain_epochs 200 --batch_size 64 --exp_name run_3ds

# Higher resolution (64³):
python main_3d.py --img_size 64 --batch_size 32 --model swin_small \
    --pretrain_epochs 200 --lr 5e-4 --exp_name run_64res
"""

import os
import sys
import argparse

import torch

from utils import get_config
from dataloader_3d import build_3d_datasets, MEDMNIST_3D_FLAGS
from engine_3d import engine_3d

sys.setrecursionlimit(40000)

# ── Dataset registry ─────────────────────────────────────────────────────────

NAME_TO_FLAG = {
    'OrganMNIST3D':   'organmnist3d',
    'NoduleMNIST3D':  'nodulemnist3d',
    'AdrenalMNIST3D': 'adrenalmnist3d',
    'FractureMNIST3D':'fracturemnist3d',
    'VesselMNIST3D':  'vesselmnist3d',
    'SynapseMNIST3D': 'synapsemnist3d',
}

ALL_3D = list(NAME_TO_FLAG.keys())

SOTA_TARGETS = {
    'OrganMNIST3D':   0.997,
    'NoduleMNIST3D':  0.863,
    'AdrenalMNIST3D': 0.874,
    'FractureMNIST3D':0.714,
    'VesselMNIST3D':  0.914,
    'SynapseMNIST3D': 0.843,
}


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Ark+ 3D-only cyclic pretraining on 6 MedMNIST 3D datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    p.add_argument('--datasets', nargs='+', default=ALL_3D,
                   choices=ALL_3D, metavar='DATASET',
                   help='Which 3D datasets to include. Default: all 6.')
    p.add_argument('--medmnist_root', default=None,
                   help='Cache dir for medmnist downloads (default ~/.medmnist)')
    p.add_argument('--img_size', type=int, default=28,
                   help='Spatial resolution of 3D volumes (28 = default MedMNIST 3D)')

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument('--model', dest='model_name', default='swin_tiny',
                   choices=['swin_tiny','swin_small','swin_base','swin_large'],
                   help='Backbone architecture')
    p.add_argument('--pretrained_weights', default=None,
                   help='Path or URL to pretrained backbone .pth / .pth.tar')
    p.add_argument('--projector_features', type=int, default=None,
                   help='Projection head output dim (default = encoder dim)')
    p.add_argument('--reinit_heads', action='store_true', default=False,
                   help='Re-init task heads when resuming (useful for fine-tuning)')

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument('--pretrain_epochs', type=int, default=200)
    p.add_argument('--batch_size',      type=int, default=64)
    p.add_argument('--workers',         type=int, default=4)
    p.add_argument('--device',          default='cuda')
    p.add_argument('--exp_name',        default='exp01')
    p.add_argument('--mode',            default='train', choices=['train','test'])
    p.add_argument('--resume',          action='store_true', default=False)
    p.add_argument('--test_epoch',      type=int, default=5,
                   help='Evaluate on test set every N epochs')
    p.add_argument('--val_loss_metric', default='average',
                   help='"average" or a dataset name to use as LR watch metric')

    # ── EMA ──────────────────────────────────────────────────────────────────
    p.add_argument('--ema_mode', default='epoch',
                   choices=['epoch','iteration'])
    p.add_argument('--momentum_teacher', type=float, default=0.9,
                   help='Initial EMA momentum (0.9 = original Ark+, '
                        'cosine-ramps to 1.0). '
                        'Use 0.9 for 3D small-dataset runs — '
                        'avoids noisy teacher at start.')

    # ── Optimiser ────────────────────────────────────────────────────────────
    p.add_argument('--opt',          default='adamw')
    p.add_argument('--opt_eps',      type=float, default=1e-8)
    p.add_argument('--opt_betas',    type=float, nargs='+', default=None)
    p.add_argument('--clip_grad',    type=float, default=None)
    p.add_argument('--momentum',     type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.05)

    # ── LR schedule ──────────────────────────────────────────────────────────
    p.add_argument('--sched',           default='cosine')
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--min_lr',          type=float, default=1e-5)
    p.add_argument('--warmup_lr',       type=float, default=1e-6)
    p.add_argument('--warmup_epochs',   type=int,   default=10)
    p.add_argument('--cooldown_epochs', type=int,   default=10)
    p.add_argument('--decay_epochs',    type=float, default=30)
    p.add_argument('--decay_rate',      type=float, default=0.5)
    p.add_argument('--patience_epochs', type=int,   default=10)
    p.add_argument('--lr_noise',        type=float, nargs='+', default=None)
    p.add_argument('--lr_noise_pct',    type=float, default=0.67)
    p.add_argument('--lr_noise_std',    type=float, default=1.0)

    args = p.parse_args()
    args.crop_size = args.img_size   # alias used internally by models
    args.use_mlp   = False           # projector is always MLP now
    return args


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print("\n" + "="*60)
    print(" Ark+ 3D-only Cyclic Pretraining")
    print("="*60)
    print(f" Model      : {args.model_name}")
    print(f" Datasets   : {args.datasets}")
    print(f" Epochs     : {args.pretrain_epochs}")
    print(f" Batch size : {args.batch_size}")
    print(f" LR         : {args.lr}")
    print(f" Momentum   : {args.momentum_teacher}")
    print(f" Device     : {args.device}")
    print(f" Exp name   : {args.exp_name}")
    print("="*60 + "\n")

    dataset_list = args.datasets

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datasets_config_3d.yaml')
    datasets_config = get_config(cfg_path)

    for ds in dataset_list:
        if ds not in datasets_config:
            raise KeyError(f"'{ds}' not found in {cfg_path}")

    # ── Print SOTA targets ────────────────────────────────────────────────────
    print(f"{'Dataset':<20s} {'Task':<30s} {'#Train':>7s} {'SOTA':>6s}")
    print("-"*68)

    # ── Build datasets ────────────────────────────────────────────────────────
    train_list, val_list, test_list = [], [], []
    for ds_name in dataset_list:
        flag = NAME_TO_FLAG[ds_name]
        tr, vl, te = build_3d_datasets(
            flag=flag, download=True, root=args.medmnist_root)
        train_list.append(tr)
        val_list.append(vl)
        test_list.append(te)
        task = datasets_config[ds_name]['task_type']
        sota = SOTA_TARGETS.get(ds_name, 0.0)
        print(f"{ds_name:<20s} {task:<30s} {len(tr):>7d} {sota:>6.3f}")

    print()

    # ── Output dirs ───────────────────────────────────────────────────────────
    model_path  = os.path.join('./Models',  f'{args.model_name}_{args.exp_name}')
    output_path = os.path.join('./Outputs', f'{args.model_name}_{args.exp_name}')
    os.makedirs('./Models',  exist_ok=True)
    os.makedirs('./Outputs', exist_ok=True)

    # ── Launch ────────────────────────────────────────────────────────────────
    engine_3d(
        args,
        model_path, output_path,
        dataset_list, datasets_config,
        train_list, val_list, test_list,
    )


if __name__ == '__main__':
    main()
