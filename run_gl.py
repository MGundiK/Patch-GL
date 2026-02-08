#!/usr/bin/env python
# run_gl.py
"""
Main entry point for xPatch-GL experiments.

Usage:
    python run_gl.py --data ETTh1 --pred_len 96 --fusion_type gated
    python run_gl.py --data ETTh1 --pred_len 96 --use_rora 1 --rora_rank 4
    python run_gl.py --data ETTh1 --pred_len 96 --gl_variant lite
"""

import argparse
import os
import torch
import random
import numpy as np

from exp.exp_xpatch_gl import Exp_XPatchGL


def main():
    parser = argparse.ArgumentParser(description='xPatch-GL: Global-Local Patch Network')

    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='xPatchGL', help='model name')

    # Data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate, S:univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Model - xPatch-GL specific
    parser.add_argument('--gl_variant', type=str, default='standard', 
                        help='xPatch-GL variant: standard, lite')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--padding_patch', type=str, default='end', help='padding type: end or None')
    parser.add_argument('--d_model', type=int, default=None, help='model dimension (default: patch_len)')
    
    # GLBlock config
    parser.add_argument('--num_gl_blocks', type=int, default=2, help='number of GLBlocks')
    parser.add_argument('--expansion_factor', type=int, default=4, help='MLP expansion in intra-patch')
    parser.add_argument('--inter_hidden_factor', type=int, default=2, help='MLP expansion in inter-patch')
    parser.add_argument('--fusion_type', type=str, default='gated', 
                        help='fusion type: gated, add, concat')
    
    # RoRA config
    parser.add_argument('--use_rora', type=int, default=0, help='use RoRA (0 or 1)')
    parser.add_argument('--rora_rank', type=int, default=4, help='RoRA rank')
    parser.add_argument('--rora_mode', type=str, default='feature', 
                        help='RoRA mode: feature, patch, both')
    
    # Decomposition
    parser.add_argument('--ema_alpha', type=float, default=0.2, help='EMA smoothing factor')
    parser.add_argument('--use_revin', type=int, default=1, help='use RevIN (0 or 1)')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--train_only', type=bool, default=False, help='train only')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')
    
    # Other (for compatibility)
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    args = parser.parse_args()
    
    # Convert int flags to bool
    args.use_rora = bool(args.use_rora)
    args.use_revin = bool(args.use_revin)

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
        # Setting string for saving
        setting = '{}_{}_{}_sl{}_pl{}_patch{}_gl{}_fusion{}_rora{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.patch_len,
            args.num_gl_blocks,
            args.fusion_type,
            args.use_rora,
        )
        
        if args.itr > 1:
            setting += f'_itr{ii}'

        exp = Exp_XPatchGL(args)
        
        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            mse, mae = exp.test(setting)
            
            print(f'\n{"="*60}')
            print(f'Final Results: {args.data} pred_len={args.pred_len}')
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
