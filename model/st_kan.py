import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools


def init_time_grid(L):
    t = torch.linspace(0, 1, L).unsqueeze(1)
    return t  # [L, 1]


def init_trend_bases(L, n):
    # Details will be released upon publication
    return Phi_tre


def init_season_bases(L, n):
    # Details will be released upon publication
    return Phi_sea


class moving_avg(nn.Module):
    """
    Moving average with head/tail padding using first/last values.
    Ensures trend does not jump at boundaries.
    """

    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        if self.kernel_size <= 1:
            return x

        # Details will be released upon publication

        return x_smooth

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # Details will be released upon publication
    def forward(self, x):
        # Details will be released upon publication
        return res, moving_mean

class Model(nn.Module):
    """
    Decomp-guided linear decomposition:

      Input: X [B, L]

      1)
         X_sea_raw, X_tre_raw = decompsition(X)

      2)
         X_tre_hat = a_tre Φ_tre^T, 用多项式拟合 trend;
         X_sea_hat = a_sea Φ_sea^T, 用 sin/cos 拟合 seasonal;

      3)  X_hat = X_tre_hat + X_sea_hat

      Output:
        X_hat, (X_tre_hat, X_sea_hat, X_res_hat),
        (a_tre, a_sea, a_res=None), lambdas={}, alpha=None
    """

    def __init__(self, args):
        super().__init__()
        self.L = args.win_size
        self.tre_order = args.n_trend
        self.sea_order = args.n_season

        # self.decompsition = decompsition   # series_decomp / DFT_series_decomp
        self.decompsition = series_decomp(args.moving_avg)

        # 1.
        Phi_tre_init = init_trend_bases(self.L, self.tre_order)  # [L, Nt]
        Phi_sea_init = init_season_bases(self.L, self.sea_order)  # [L, Ns]

        Nt = Phi_tre_init.shape[1]
        Ns = Phi_sea_init.shape[1]

        # Details will be released upon publication

        print(
            f"Decomp-guided Model initialized: "
            f"L={self.L}, Nt={Nt}, Ns={Ns}"
        )

    def domain_params(self):

        # Details will be released upon publication

    def target_params(self):

        # Details will be released upon publication

    def forward(self, X):
        """
        X: [B, L]
        """
        B, L = X.shape
        assert L == self.L, f"Expected window length {self.L}, got {L}"

        # Details will be released upon publication

        lambdas = {
            "tre_raw": tre_raw,  # [B, L]
            "sea_raw": sea_raw,  # [B, L]
        }


        return X_hat, (X_tre_hat, X_sea_hat, X_res_hat), (a_tre, a_sea), lambdas

    def compute_basis_usage(self, a_tre, a_sea, eps=1e-8):  #
        # a_tre: [B, Nt], a_sea: [B, Ns]
        #
        usage_tre = torch.abs(a_tre)
        usage_tre = usage_tre / (usage_tre.sum(dim=1, keepdim=True) + eps)  # [B, Nt]
        usage_tre = usage_tre.mean(dim=0)  # [Nt]

        usage_sea = torch.abs(a_sea)
        usage_sea = usage_sea / (usage_sea.sum(dim=1, keepdim=True) + eps)  # [B, Ns]
        usage_sea = usage_sea.mean(dim=0)  # [Ns]

        return usage_tre, usage_sea

