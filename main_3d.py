"""
main_3d.py  —  Ark+ 3D-only cyclic pretraining

Quick start:
    python main_3d.py --exp_name run05 --pretrain_epochs 300

Resume:
    python main_3d.py --exp_name run05 --pretrain_epochs 300 --resume
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch

from utils import get_config
from dataloader_3d import build_3d_datasets
from engine_3d import engine_3d

sys.setrecursionlimit(40000)

NAME_TO_FLAG = {
    'OrganMNIST3D':   'organmnist3d',
    'NoduleMNIST3D':  'nodulemnist3d',
    'AdrenalMNIST3D': 'adrenalmnist3d',
    'FractureMNIST3D':'fracturemnist3d',
    'VesselMNIST3D':  'vesselmnist3d',
    'SynapseMNIST3D': 'synapsemnist3d',
}
ALL_3D = list(NAME_TO_FLAG.keys())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',        nargs='+', default=ALL_3D, choices=ALL_3D)
    p.add_argument('--medmnist_root',   default=None)
    p.add_argument('--img_size',        type=int,   default=28)
    p.add_argument('--model',           dest='model_name', default='swin_tiny',
                   choices=['swin_tiny','swin_small','swin_base','swin_large'])
    p.add_argument('--pretrain_epochs', type=int,   default=300)
    p.add_argument('--batch_size',      type=int,   default=128)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--weight_decay',    type=float, default=0.1)     # was 0.05
    p.add_argument('--warmup_epochs',   type=int,   default=10)
    p.add_argument('--workers',         type=int,   default=8)
    p.add_argument('--device',          default='cuda')
    p.add_argument('--exp_name',        default='run05')
    p.add_argument('--test_epoch',      type=int,   default=10)
    p.add_argument('--resume',          action='store_true', default=False)
    p.add_argument('--pretrained_weights', default=None)
    p.add_argument('--projector_features', type=int, default=None)
    args = p.parse_args()
    args.crop_size = args.img_size
    return args


def main():
    args = get_args()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datasets_config_3d.yaml')
    datasets_config = get_config(cfg_path)

    train_list, val_list, test_list = [], [], []
    for ds_name in args.datasets:
        flag = NAME_TO_FLAG[ds_name]
        tr, vl, te = build_3d_datasets(flag=flag, download=True,
                                       root=args.medmnist_root)
        train_list.append(tr)
        val_list.append(vl)
        test_list.append(te)

    model_path  = os.path.join('Models',  args.exp_name)
    output_path = os.path.join('Outputs', args.exp_name)

    engine_3d(
        args,
        model_path, output_path,
        args.datasets, datasets_config,
        train_list, val_list, test_list,
    )


if __name__ == '__main__':
    main()
