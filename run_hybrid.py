#!/usr/bin/env python
"""
xPatch-GL Hybrid: Main Entry Point

Usage:
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant full
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_agg
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_ms
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant no_gate
    python run_hybrid.py --data ETTh1 --pred_len 96 --variant baseline  # Pure xPatch-style
"""

import argparse
import os
import torch
import random
import numpy as np

from exp.exp_xpatch_gl_hybrid import Exp_xPatchGL_Hybrid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='xPatch-GL Hybrid')
    
    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model_id', type=str, default='xPatchGL_Hybrid', help='model id')
    parser.add_argument('--model', type=str, default='xPatchGL_Hybrid', help='model name')
    
    # Data config
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='M/S/MS')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints path')
    
    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    
    # Model config
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    
    # xPatch config
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
    
    # Training config
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--lradj', type=str, default='type1', help='lr adjustment type')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision')
    
    # GPU config
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multi gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='gpu ids')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Parse kernel sizes
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    # Apply variant shortcuts
    if args.variant == 'full':
        args.use_aggregate_conv = True
        args.use_multiscale = True
        args.use_gating = True
    elif args.variant == 'no_agg':
        args.use_aggregate_conv = False
        args.use_multiscale = True
        args.use_gating = True
    elif args.variant == 'no_ms':
        args.use_aggregate_conv = True
        args.use_multiscale = False
        args.use_gating = True
    elif args.variant == 'no_gate':
        args.use_aggregate_conv = True
        args.use_multiscale = True
        args.use_gating = False
    elif args.variant == 'baseline':
        args.use_aggregate_conv = False
        args.use_multiscale = False
        args.use_gating = False
    
    # Convert int flags to bool
    args.use_aggregate_conv = bool(args.use_aggregate_conv)
    args.use_multiscale = bool(args.use_multiscale)
    args.use_gating = bool(args.use_gating)
    
    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)
    
    # Create setting string
    setting = f'xPatchGL_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'
    setting += f'_agg{int(args.use_aggregate_conv)}_ms{int(args.use_multiscale)}_gate{int(args.use_gating)}'
    
    # Run experiment
    exp = Exp_xPatchGL_Hybrid(args)
    
    if args.is_training:
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        mse, mae = exp.test(setting)
        
        print('\n' + '=' * 60)
        print(f'Final Results: {args.data} pred_len={args.pred_len}')
        print(f'MSE: {mse:.6f}')
        print(f'MAE: {mae:.6f}')
        print('=' * 60)
        
        torch.cuda.empty_cache()
    else:
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
