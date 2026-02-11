#!/usr/bin/env python
# run_hybrid.py
"""
Main entry point for xPatch-GL Hybrid experiments.

Usage:
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant full
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_agg
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_ms
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_gate
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant baseline
"""

import argparse
import os
import torch
import random
import numpy as np

from exp.exp_xpatch_gl_hybrid import Exp_xPatchGL_Hybrid


def main():
    parser = argparse.ArgumentParser(description='xPatch-GL Hybrid')

    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='xPatchGL_Hybrid', help='model name')

    # Data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Model - xPatch config
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--ema_alpha', type=float, default=0.3, help='EMA smoothing factor')
    
    # GLCN enhancement flags
    parser.add_argument('--use_aggregate_conv', type=int, default=1, help='use aggregate conv before patching')
    parser.add_argument('--use_multiscale', type=int, default=1, help='use multi-scale depthwise conv')
    parser.add_argument('--use_gating', type=int, default=1, help='use gated pointwise conv')
    parser.add_argument('--kernel_sizes', type=str, default='1,3,5,7', help='multi-scale kernel sizes')
    
    # Variant shortcut
    parser.add_argument('--variant', type=str, default='full', 
                        choices=['full', 'no_agg', 'no_ms', 'no_gate', 'baseline'],
                        help='model variant')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='sigmoid', help='adjust learning rate')
    parser.add_argument('--train_only', type=bool, default=False, help='train only')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')
    
    # For compatibility with data_provider
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    args = parser.parse_args()
    
    # Parse kernel sizes
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    # Apply variant shortcuts
    if args.variant == 'full':
        args.use_aggregate_conv = 1
        args.use_multiscale = 1
        args.use_gating = 1
    elif args.variant == 'no_agg':
        args.use_aggregate_conv = 0
        args.use_multiscale = 1
        args.use_gating = 1
    elif args.variant == 'no_ms':
        args.use_aggregate_conv = 1
        args.use_multiscale = 0
        args.use_gating = 1
    elif args.variant == 'no_gate':
        args.use_aggregate_conv = 1
        args.use_multiscale = 1
        args.use_gating = 0
    elif args.variant == 'baseline':
        args.use_aggregate_conv = 0
        args.use_multiscale = 0
        args.use_gating = 0
    
    # Convert int flags to bool
    args.use_aggregate_conv = bool(args.use_aggregate_conv)
    args.use_multiscale = bool(args.use_multiscale)
    args.use_gating = bool(args.use_gating)

    # Set device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Run experiments
    for ii in range(args.itr):
        # Get actual dataset name (extract from data_path if data='custom')
        if args.data == 'custom':
            dataset_name = args.data_path.replace('.csv', '')
        else:
            dataset_name = args.data
        
        # Setting string for saving
        setting = '{}_{}_{}_sl{}_pl{}_agg{}_ms{}_gate{}'.format(
            args.model,
            dataset_name,
            args.features,
            args.seq_len,
            args.pred_len,
            int(args.use_aggregate_conv),
            int(args.use_multiscale),
            int(args.use_gating),
        )
        
        if args.itr > 1:
            setting += f'_itr{ii}'

        exp = Exp_xPatchGL_Hybrid(args)
        
        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            mse, mae = exp.test(setting)
            
            print(f'\n{"="*60}')
            print(f'Final Results: {dataset_name} pred_len={args.pred_len}')
            print(f'MSE: {mse:.6f}')
            print(f'MAE: {mae:.6f}')
            print(f'{"="*60}\n')
            
            torch.cuda.empty_cache()
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    main()
