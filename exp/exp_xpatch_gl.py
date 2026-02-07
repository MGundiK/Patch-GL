# exp/exp_xpatch_gl.py
"""
Experiment class for xPatch-GL

Integrates with the existing data loading infrastructure and training loop.
Based on exp_main.py but adapted for xPatch-GL.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')


class Exp_XPatchGL(Exp_Basic):
    def __init__(self, args):
        super(Exp_XPatchGL, self).__init__(args)

    def _build_model(self):
        # Import the model
        from layers.network_gl import NetworkGL, NetworkGL_Lite
        
        # Determine variant
        variant = getattr(self.args, 'gl_variant', 'standard')
        
        if variant == 'lite':
            model = NetworkGL_Lite(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                enc_in=self.args.enc_in,
                patch_len=getattr(self.args, 'patch_len', 16),
                stride=getattr(self.args, 'stride', 8),
                expansion_factor=getattr(self.args, 'expansion_factor', 2),
                dropout=getattr(self.args, 'dropout', 0.1),
                ema_alpha=getattr(self.args, 'ema_alpha', 0.2),
            ).float()
        else:
            model = NetworkGL(
                # Core dimensions
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                enc_in=self.args.enc_in,
                # Patching
                patch_len=getattr(self.args, 'patch_len', 16),
                stride=getattr(self.args, 'stride', 8),
                padding_patch=getattr(self.args, 'padding_patch', 'end'),
                # Model dimensions
                d_model=getattr(self.args, 'd_model', None),
                num_gl_blocks=getattr(self.args, 'num_gl_blocks', 2),
                expansion_factor=getattr(self.args, 'expansion_factor', 4),
                inter_hidden_factor=getattr(self.args, 'inter_hidden_factor', 2),
                fusion_type=getattr(self.args, 'fusion_type', 'gated'),
                dropout=getattr(self.args, 'dropout', 0.1),
                # RoRA
                use_rora=getattr(self.args, 'use_rora', False),
                rora_rank=getattr(self.args, 'rora_rank', 4),
                rora_mode=getattr(self.args, 'rora_mode', 'feature'),
                # Decomposition
                ema_alpha=getattr(self.args, 'ema_alpha', 0.2),
                # RevIN
                use_revin=getattr(self.args, 'use_revin', True),
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # xPatch-GL forward pass (only needs batch_x)
                outputs = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Apply loss weighting if not test
                if not is_test:
                    # Arctangent loss with weight decay
                    self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to(self.device)
                    pred = outputs * self.ratio
                    true = batch_y * self.ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # xPatch-GL forward pass
                outputs = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Arctangent loss with weight decay
                self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to(self.device)

                outputs = outputs * self.ratio
                batch_y = batch_y * self.ratio

                loss = mae_criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # xPatch-GL forward pass
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return mse, mae
