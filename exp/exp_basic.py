import os

import pandas as pd
import torch
from model import st_kan
import matplotlib.pyplot as plt


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'STKAN': st_kan,
        }
        self.save_file_setting = (f'mutli{args.is_domain}_bs{args.batch_size}_ws{args.win_size}_'
                                  f'tr{args.n_trend}_se{args.n_season}_re{args.n_residual}_'
                                  f'_')
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                print('Use MPS')
                return device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
