import argparse
import torch

from exp.exp_2phase_dom_anomaly_detection import Exp_TwoPhase_Dom_Anomaly_Detection

import random
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MultiDatasetAD')

# basic config

parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='STKAN',
                    help='model name, options: [STKAN]')

# model info
parser.add_argument("--dropout", default=0.1, type=float)

parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--n_trend', type=int, default=3, help='num of trend bases')
parser.add_argument('--n_season', type=int, default=20, help='num of season bases')

# data loader
parser.add_argument('--data', type=str, required=False, default='TSB_U', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/TSB_AD/TSB_AD_U', help='root path of the data file')
parser.add_argument('--file_class', type=str, required=False, default='SMD', help='file class name (only for UCR)')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--save_paras_path', type=str, default='./save_paras/', help='location of the saved parameters for visual')


# multi-dataset setting
parser.add_argument('--is_domain', type=str2bool, default=True, help='if use multi dataset in domain')

# data info
parser.add_argument("--moving_avg", default=101, type=int)
parser.add_argument("--win_size", "-ws", required=False, default=128, type=int)
parser.add_argument('--feature_size', type=int, default=17, help='feature size')
parser.add_argument("--device", required=False, default='cuda:0', type=str)
parser.add_argument("--ver", required=False, default='', type=str)
parser.add_argument("--stride", "-st", required=False, default=1, type=int)
parser.add_argument("--batch_size", "-bs", required=False, default=16, type=int)
parser.add_argument("--epochs", '-ep', required=False, default=15, type=int)

# anomaly detection task

parser.add_argument("--exp_type", type=str, default='2phase')

# training paras
parser.add_argument("--use_phase1", required=False, default=True, type=str2bool)
parser.add_argument("--use_phase2", required=False, default=True, type=str2bool)
parser.add_argument('--phase1_epochs', type=int, default=7, help='num of phase1 epochs')
parser.add_argument('--phase2_epochs', type=int, default=10, help='num of phase2 epochs')

# optimize basis
parser.add_argument('--usage_reg_lambda', type=float, default=0.01, help='usage regularization for base')
parser.add_argument('--usage_high_th', type=float, default=0.4, help='usage threshold for base')
parser.add_argument('--residual_ft_th', type=float, default=0.01, help='residual threshold for base retrain in phase2')

# Base
parser.add_argument('--base_ft_epochs', type=int, default=1, help='num of base finetune epochs')
parser.add_argument('--base_ft_lr_scale', type=float, default=0.01, help='base finetune learning rate scale')

# Phase1
parser.add_argument('--base_reg_lambda', type=float, default=0.1, help='L2 regularization to keep bases close to init')
parser.add_argument('--smooth_reg_lambda', type=float, default=0.0, help='smooth regularization for trend bases')


# ---------------- Phase 2 (Target-Specific Calibration) ----------------
parser.add_argument("--use_phase2", type=str2bool, default=True)

# Stage 1: head-only
parser.add_argument("--phase2_min_head_epochs", type=int, default=3)

# Trigger thresholds (Eq.(trigger) in the paper)
parser.add_argument("--phase2_residual_gap_ratio_th", type=float, default=0.95)
parser.add_argument("--phase2_kl_th", type=float, default=0.01)
parser.add_argument("--phase2_rare_ratio_th", type=float, default=5.0)
parser.add_argument("--phase2_rare_global_th", type=float, default=0.05)

# Stage 2: basis refinement (optional)
parser.add_argument("--phase2_update_base", type=str2bool, default=True)
parser.add_argument("--base_ft_epochs", type=int, default=1)
parser.add_argument("--base_ft_lr_scale", type=float, default=0.1)
parser.add_argument("--base_prox_lambda", type=float, default=0.0)
parser.add_argument("--phase2_tunable_usage_th", type=float, default=0.02)

# Stage 3: head refit
parser.add_argument("--head_refit_epochs", type=int, default=2)


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='none', help='com')
parser.add_argument('--grid_search', type=bool, default=True)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)


Exp = Exp_TwoPhase_Dom_Anomaly_Detection




if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}_{}'.format(
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.file_class,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test_mine(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_{}_{}'.format(
        args.model_id,
        args.comment,
        args.model,
        args.data,
        args.file_class,
        ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test_mine(setting, test=1)
    torch.cuda.empty_cache()
