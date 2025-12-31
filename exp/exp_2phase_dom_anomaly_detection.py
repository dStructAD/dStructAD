from cmath import phase

from einops import rearrange
from numpy.ma.core import reshape
from torch.optim import lr_scheduler

# from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils.evaluate_tools import find_best_threshold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from data_provider.data_factory import multi_data_provider
from torch.utils.data import DataLoader, ConcatDataset, Dataset


torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class LabeledDataset(Dataset):
    def __init__(self, base_dataset, dataset_label):
        """
        base_dataset: 原始 PyTorch dataset
        dataset_label: 整数标签，用来标识数据来源
        """
        self.base_dataset = base_dataset
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return (self.base_dataset[idx], self.dataset_label)


class Exp_TwoPhase_Dom_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_TwoPhase_Dom_Anomaly_Detection, self).__init__(args)
        self.save_tsne = True

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):

        self.file_names, all_datasets = multi_data_provider(self.args, flag)
        select_data_files = {}

        # ----------------#
        # in this section, only current data need to be used
        # each dataset, each dataloader
        if flag == 'test' or flag == 'thre' or flag == 'test_window':
            shuffle_flag = False
            batch_size = self.args.batch_size
            data_set = []
            for current_file in self.file_names:
                data_set.append(all_datasets[current_file])

            data_set = ConcatDataset(data_set)
            data_loader = [
                DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
                for ds in data_set.datasets
            ]
            return data_loader

        elif flag == 'train_thre':
            shuffle_flag = False
            batch_size = self.args.batch_size
            self.class_file_lengths = {}
            data_loader = []
            for current_file in self.file_names:
                data_set = []
                data_set.append(LabeledDataset(all_datasets[current_file], current_file[:3]))
                self.class_file_lengths[current_file] = len(all_datasets[current_file])
                if self.args.is_domain:  # hyper for ablation -- multidatasets
                    if self.args.select_auxset:
                        for other_file in select_data_files[current_file]:
                            data_set.append(LabeledDataset(all_datasets[other_file], other_file[:3]))
                    else:
                        for other_file in self.file_names:
                            if other_file == current_file:
                                continue
                            data_set.append(LabeledDataset(all_datasets[other_file], other_file[:3]))
                data_set = ConcatDataset(data_set)
                current_data_dl = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag)

                data_loader.append(current_data_dl)

            return data_loader

        # training, validation
        # all dataloader include two loader, ALl Loader and Target Loader
        elif flag == 'train' or flag == 'val':
            batch_size = self.args.batch_size
            shuffle_flag = True
            drop_last = False

            domain_loaders = []
            target_loaders = []

            for current_file in self.file_names:
                target_set = LabeledDataset(all_datasets[current_file], 1)
                target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=shuffle_flag,
                                           num_workers=self.args.num_workers, drop_last=drop_last)
                target_loaders.append(target_loader)


                if self.args.is_domain:  # hyper for ablation -- multidatasets
                    domain_sets = []
                    for other_file in self.file_names:
                        if other_file == current_file:
                            continue
                        domain_sets.append(LabeledDataset(all_datasets[other_file], 0))
                    domain_sets = ConcatDataset(domain_sets)
                    domain_data_loader = DataLoader(domain_sets, batch_size=batch_size, shuffle=shuffle_flag,
                                                    num_workers=self.args.num_workers, drop_last=drop_last)

                    domain_loaders.append(domain_data_loader)

                print(current_file)

            return target_loaders, domain_loaders

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    @torch.no_grad()
    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        criterion = self._select_criterion()

        for i, (batch_x, dataset_label) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)

            X_hat, (Xt, Xs, Xr), (at, as_, ar), lambdas, alpha = self.model(batch_x)
            rec_loss = criterion(X_hat, batch_x)

            loss_1.append(rec_loss.item())
            loss_2.append(rec_loss.item())

        self.model.train()
        return np.average(loss_1), np.average(loss_2)

    def train_dom_single(self, setting, train_loader, vali_loader, dom_loader):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.is_domain:
            best_model_path = path + '/' + self.save_file_setting + self.file_now + 'checkpoint.pth'
        else:
            best_model_path = path + '/' + self.save_file_setting + self.file_now + '_checkpoint.pth'

        time_now = time.time()

        target_train_steps = len(train_loader)
        domain_train_steps = len(dom_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        domain_optimizer = optim.Adam(self.model.domain_params(), lr=self.args.learning_rate)
        target_optimizer = optim.Adam(self.model.target_params(), lr=self.args.learning_rate)

        criterion = self._select_criterion()

        domain_scheduler = lr_scheduler.OneCycleLR(optimizer=domain_optimizer,
                                                   steps_per_epoch=domain_train_steps,
                                                   pct_start=self.args.pct_start,
                                                   epochs=self.args.train_epochs,
                                                   max_lr=self.args.learning_rate)

        target_scheduler = lr_scheduler.OneCycleLR(optimizer=target_optimizer,
                                                   steps_per_epoch=target_train_steps,
                                                   pct_start=self.args.pct_start,
                                                   epochs=self.args.train_epochs,
                                                   max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count_dom = 0
            train_loss_dom = []

            iter_count_tgt = 0
            train_loss_tgt = []

            self.model.train()
            epoch_time = time.time()


            for i, (batch_x, dataset_label) in enumerate(dom_loader):
                iter_count_dom += 1
                domain_optimizer.zero_grad()
                B, L = batch_x.shape
                batch_x = batch_x.float().to(self.device)

                X_hat, (Xt, Xs, Xr), (at, as_), lambdas = self.model(batch_x)

                # 0) 重构 loss
                loss = criterion(X_hat, batch_x)

                train_loss_dom.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("domain \titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count_dom
                    left_time = speed * ((self.args.train_epochs - epoch) * domain_train_steps - i)
                    print('domain \tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count_dom = 0
                    time_now = time.time()

                loss.backward()
                domain_optimizer.step()
                # time.sleep(0.1)
                if self.args.lradj == 'TST':
                    adjust_learning_rate(domain_optimizer, domain_scheduler, epoch + 1, self.args, printout=False)
                    domain_scheduler.step()


            for i, (batch_x, dataset_label) in enumerate(train_loader):
                iter_count_tgt += 1
                target_optimizer.zero_grad()
                B, L = batch_x.shape
                batch_x = batch_x.float().to(self.device)

                X_hat, (Xt, Xs, Xr), (at, as_), lambdas = self.model(batch_x)

                # 0) 重构 loss
                loss = criterion(X_hat, batch_x)

                train_loss_tgt.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(" target \titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count_tgt
                    left_time = speed * ((self.args.train_epochs - epoch) * target_train_steps - i)
                    print(' target \tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count_tgt = 0
                    time_now = time.time()

                loss.backward()
                target_optimizer.step()
                # time.sleep(0.1)
                if self.args.lradj == 'TST':
                    adjust_learning_rate(target_optimizer, target_scheduler, epoch + 1, self.args, printout=False)
                    target_scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss_tgt)
            vali_loss1, vali_loss2 = self.vali(vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss1: {3:.7f} Vali Loss2: {3:.7f}".format(
                epoch + 1, target_train_steps, train_loss, vali_loss1, vali_loss2))
            early_stopping(vali_loss1, vali_loss2, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(domain_optimizer, domain_scheduler, epoch + 1, self.args, printout=True)
                adjust_learning_rate(target_optimizer, target_scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(target_scheduler.get_last_lr()[0]))

    def train_phase1_base(self, setting, tgt_train_loaders, tgt_vali_loaders):

        print("==== Phase 1: Pretrain shared bases on all TargetSets ====")

        # checkpoint
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        phase1_stats_model_path = os.path.join(path, self.save_file_setting + 'phase1_base.pth')

        # ----  freeze target_params，train domain_params ----
        for p in self.model.target_params():
            p.requires_grad = False
        for p in self.model.domain_params():
            p.requires_grad = True

        optimizer = optim.Adam(self.model.domain_params(), lr=self.args.learning_rate)
        recon_criterion = nn.MSELoss().to(self.device)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        time_now = time.time()

        # 记录 epoch 级别的 loss 和 usage
        phase1_train_loss_log = []
        phase1_vali_loss1_log = []
        phase1_vali_loss2_log = []
        phase1_usage_tre_log = []  # list of [Nt] numpy
        phase1_usage_sea_log = []  # list of [Ns] numpy

        # 全局 usage 统计（跨 epoch）
        global_usage_tre_sum = None
        global_usage_sea_sum = None
        global_usage_count = 0

        for epoch in range(self.args.phase1_epochs):
            self.model.train()
            epoch_time = time.time()
            iter_count = 0
            train_losses = []

            # 本 epoch 的 usage 累积
            epoch_usage_tre_sum = None
            epoch_usage_sea_sum = None
            epoch_usage_count = 0

            # 依次遍历每个 target 的 train_loader
            for file_idx, train_loader in enumerate(tgt_train_loaders):
                current_file = self.file_names[file_idx]
                print(f'[Phase1] training on target file: {current_file}')

                for i, (batch_x, dataset_label) in enumerate(train_loader):
                    iter_count += 1
                    optimizer.zero_grad()

                    B, L = batch_x.shape
                    batch_x = batch_x.float().to(self.device)

                    X_hat, (Xt, Xs, Xr), (at, as_), _ = self.model(batch_x)

                    recon_loss = recon_criterion(X_hat, batch_x)

                    # ---------- Regularization ----------
                    loss = recon_loss

                    #
                    usage_tre, usage_sea = self.model.compute_basis_usage(at, as_)

                    # (1) u
                    if getattr(self.args, 'usage_reg_lambda', 0.0) > 0:
                        high_th = getattr(self.args, 'usage_high_th', 0.4)
                        over_tre = F.relu(usage_tre - high_th).sum()
                        over_sea = F.relu(usage_sea - high_th).sum()
                        usage_reg = over_tre + over_sea
                        loss = loss + self.args.usage_reg_lambda * usage_reg

                    # (2)
                    base_reg_lambda = getattr(self.args, 'base_reg_lambda', 0.1)
                    if base_reg_lambda > 0:
                        base_reg = F.mse_loss(self.model.Phi_tre, self.model.Phi_tre_init) + \
                                   F.mse_loss(self.model.Phi_sea, self.model.Phi_sea_init)
                        loss = loss + base_reg_lambda * base_reg


                    # 累加到 epoch 统计
                    if epoch_usage_tre_sum is None:
                        epoch_usage_tre_sum = usage_tre.detach().cpu()
                        epoch_usage_sea_sum = usage_sea.detach().cpu()
                    else:
                        epoch_usage_tre_sum += usage_tre.detach().cpu()
                        epoch_usage_sea_sum += usage_sea.detach().cpu()
                    epoch_usage_count += 1

                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    if (i + 1) % 1000 == 0:
                        print("[Phase1] file: {0} | iters: {1}, epoch: {2} | loss: {3:.7f}".format(
                            current_file, i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / max(1, iter_count)
                        left_time = speed * (
                                (self.args.phase1_epochs - epoch - 1) * len(train_loader) + (len(train_loader) - i - 1)
                        )
                        print('[Phase1] speed: {:.4f}s/iter; estimated left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            # —— ——
            if epoch_usage_count > 0:
                epoch_usage_tre = (epoch_usage_tre_sum / epoch_usage_count).numpy()
                epoch_usage_sea = (epoch_usage_sea_sum / epoch_usage_count).numpy()
                phase1_usage_tre_log.append(epoch_usage_tre)
                phase1_usage_sea_log.append(epoch_usage_sea)

                if global_usage_tre_sum is None:
                    global_usage_tre_sum = epoch_usage_tre_sum.clone()
                    global_usage_sea_sum = epoch_usage_sea_sum.clone()
                else:
                    global_usage_tre_sum += epoch_usage_tre_sum
                    global_usage_sea_sum += epoch_usage_sea_sum
                global_usage_count += epoch_usage_count

            #  train loss
            train_loss = np.average(train_losses) if len(train_losses) > 0 else 0.0
            phase1_train_loss_log.append(train_loss)

            # vali_loader
            vali_losses1, vali_losses2 = [], []
            for v_loader in tgt_vali_loaders:
                v1, v2 = self.vali(v_loader)
                vali_losses1.append(v1)
                vali_losses2.append(v2)
            vali_loss1 = float(np.average(vali_losses1)) if len(vali_losses1) > 0 else 0.0
            vali_loss2 = float(np.average(vali_losses2)) if len(vali_losses2) > 0 else 0.0
            phase1_vali_loss1_log.append(vali_loss1)
            phase1_vali_loss2_log.append(vali_loss2)

            print("Phase1 Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            print("Phase1 Epoch: {0} | Train Loss: {1:.7f} | Vali Loss1: {2:.7f} | Vali Loss2: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss1, vali_loss2))

            early_stopping(vali_loss1, vali_loss2, self.model, phase1_stats_model_path)
            if early_stopping.early_stop:
                print("Phase1 Early stopping triggered.")
                break




    def train_phase2_single_target(self, setting, train_loader, vali_loader, target_name):
        """
        Phase 2 (Target-Specific Calibration) for a single TargetSet.

        Workflow:
          Stage 1 (head-only): optimize TargetSet-specific parameters theta_n with the domain basis frozen.
          Stage 2 (optional base): if the TargetSet cannot be explained by coefficient adaptation alone,
                                  selectively fine-tune the domain basis under structure-preserving constraints.
          Stage 3 (head refit): re-optimize theta_n after the basis update.

        This implementation is intentionally "review-friendly":
          - minimal logging,
          - English comments,
          - explicit hyperparameters.
        """
        print(f"==== Phase 2: Target-specific calibration for {target_name} ====")
        self.file_now = target_name

        # ---------------------------------------------------------------------
        # Paths / checkpoint
        # ---------------------------------------------------------------------
        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)

        if getattr(self.args, "is_domain", False):
            best_model_path = os.path.join(ckpt_dir, f"{self.save_file_setting}{target_name}checkpoint.pth")
        else:
            best_model_path = os.path.join(ckpt_dir, f"{self.save_file_setting}checkpoint.pth")

        criterion = nn.MSELoss()

        # ---------------------------------------------------------------------
        # Hyperparameters (must be defined in args for reproducibility)
        # ---------------------------------------------------------------------
        lr = self.args.learning_rate
        head_epochs = self.args.phase2_min_head_epochs
        base_epochs = self.args.base_ft_epochs
        head_refit_epochs = self.args.head_refit_epochs

        # Trigger thresholds
        residual_ratio_th = self.args.phase2_residual_gap_ratio_th
        kl_th = self.args.phase2_kl_th
        rare_ratio_th = self.args.phase2_rare_ratio_th
        rare_global_th = self.args.phase2_rare_global_th

        # Whether to ever update the basis in Phase 2
        enable_base_update = self.args.phase2_update_base

        # Structure-preserving constraints for basis refinement
        base_ft_lr_scale = getattr(self.args, "base_ft_lr_scale", 0.1)
        prox_lambda = self.args.base_prox_lambda  # weighted proximal term to the Phase-1 anchor
        tunable_usage_th = self.args.phase2_tunable_usage_th  # selective backward: only update low-usage bases

        # ---------------------------------------------------------------------
        # (0) Phase-1 baseline reconstruction loss on this TargetSet (frozen Phase-1 model)
        #     Used to compute residual-gap ratio for the Phase-2 trigger.
        # ---------------------------------------------------------------------
        self.model.eval()
        phase1_loss_sum, phase1_loss_cnt = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                x_hat, *_ = self.model(batch_x)
                phase1_loss_sum += criterion(x_hat, batch_x).item()
                phase1_loss_cnt += 1
        phase1_baseline_loss = phase1_loss_sum / max(phase1_loss_cnt, 1)
        print(f"[Phase2] Phase-1 baseline recon loss on {target_name}: {phase1_baseline_loss:.6f}")

        # Global usage statistics (learned in Phase 1)

        # Details will be released upon publication
        if not has_global_usage:
            print("[Phase2] global_usage_* is missing: KL / rare-mode diagnostics will be skipped.")

        # ---------------------------------------------------------------------
        # Stage 1: head-only personalization (freeze domain params, train target params)
        # ---------------------------------------------------------------------

        # Details will be released upon publication

        # Restore the best head-only checkpoint
        if best_head_state is not None:
            self.model.load_state_dict(best_head_state, strict=True)

        # ---------------------------------------------------------------------
        # Decide whether we need basis refinement (Stage 2)
        # ---------------------------------------------------------------------
        need_base_ft = False
        residual_gap_ratio = float("nan")

        # Details will be released upon publication

        if not need_base_ft:
            torch.save(self.model.state_dict(), best_model_path)
            print(f"[Phase2] saved head-only model: {best_model_path}")
            return

        # ---------------------------------------------------------------------
        # Stage 2: basis refinement (freeze target params, update domain params)
        #   - optional weighted proximal term to the Phase-1 anchor
        #   - selective backward: update only low-usage bases
        # ---------------------------------------------------------------------
        print(f"[Phase2-Stage2] refining domain basis for {target_name}")

        # Cache the anchor basis for proximal regularization if available
        global_Phi_tre = getattr(self, "global_Phi_tre", None)
        global_Phi_sea = getattr(self, "global_Phi_sea", None)

        # Details will be released upon publication

        with torch.no_grad():
            Phi_tre_before = self.model.Phi_tre.detach().cpu().clone()
            Phi_sea_before = self.model.Phi_sea.detach().cpu().clone()

        for epoch in range(base_epochs):
            self.model.train()
            train_losses = []

            for batch_x, _ in train_loader:
                batch_x = batch_x.float().to(self.device)

                base_optim.zero_grad()
                x_hat, _, _, _, _ = self.model(batch_x)
                loss = criterion(x_hat, batch_x)

                # Weighted proximal to the Phase-1 anchor (higher-usage bases are more constrained)
                if prox_lambda > 0 and (global_Phi_tre is not None) and (global_Phi_sea is not None) and (
                        w_tre is not None) and (w_sea is not None):
                # Details will be released upon publication

                loss.backward()

                # Selective backward: only update low-usage bases
                if (w_tre is not None) and (w_sea is not None) and (tunable_usage_th is not None):
                    with torch.no_grad():
                    # Details will be released upon publication

                base_optim.step()
                train_losses.append(loss.item())

            vali_loss1, _ = self.vali(vali_loader)
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            print(f"[Phase2-Stage2] epoch {epoch + 1}/{base_epochs} | train={train_loss:.6f} | vali={vali_loss1:.6f}")

        # Basis shift (optional diagnostic)
        with torch.no_grad():

            # Details will be released upon publication
        print(
            f"[Phase2-Stage2] basis shift | tre(mean)={basis_shift_tre.mean():.6f} | sea(mean)={basis_shift_sea.mean():.6f}")



        # Details will be released upon publication


    def train(self, setting):

        #
        tgt_train_loaders, dom_train_loaders = self._get_data(flag='train')
        tgt_vali_loaders, dom_vali_loaders = self._get_data(flag='val')

        # Phase1:
        if getattr(self.args, 'use_phase1', True):
            self.train_phase1_base(setting, tgt_train_loaders, tgt_vali_loaders)

        # Phase2:
        if getattr(self.args, 'use_phase2', True):
            for i, (train_loader, vali_loader) in enumerate(zip(tgt_train_loaders, tgt_vali_loaders)):
                current_file = self.file_names[i]
                print(f'==== Phase 2 start: current target file is {current_file} ====')
                self.train_phase2_single_target(setting, train_loader, vali_loader, current_file)


    @torch.no_grad()
    def test_mine(self, setting, test=0):
        """
        """
        # 1)
        thre_loaders = self._get_data(flag='test_window')
        train_thre_loaders = self._get_data(flag='train_thre')

        folder_path = './gridsearch_test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        all_result = []
        self.output_folder = folder_path

        print("===== Start testing for setting: {} =====".format(setting))
        print("Total target files for test:", len(self.file_names))

        for i, (test_loader, train_loader) in enumerate(zip(thre_loaders, train_thre_loaders)):
            self.file_now = self.file_names[i]
            print('\n[TEST] current file index: {} | name: {}'.format(i, self.file_now))

            if test:
                print('  -> loading model checkpoint for this file')
                path = os.path.join(self.args.checkpoints, setting)
                if self.args.is_domain:
                    best_model_path = os.path.join(
                        path, self.save_file_setting + self.file_now + 'checkpoint.pth'
                    )
                else:
                    best_model_path = os.path.join(
                        path, self.save_file_setting + self.file_now + '_checkpoint.pth'
                    )

                print('     checkpoint path:', best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()

            #
            self._compute_global_usage_from_loaders(train_loader)

            #
            current_result = self.test_single(test_loader, i)
            all_result.append(current_result)

        all_result = np.stack(all_result)
        print("\n===== All files tested. Now aggregating results. =====")
        self.save_avg_result(all_result)
        print("===== Testing done. Results saved to: {} =====".format(self.output_folder))

    @torch.no_grad()
    def _compute_global_usage_from_loaders(self, loader):
        self.model.eval()
        tre_sum, sea_sum, cnt = None, None, 0
        # Details will be released upon publication

    @torch.no_grad()
    def test_single(self, test_loader_single, index, calib=None):
        """
        对单个文件进行异常检测评估：
          - 输入：该文件的 test_window dataloader
          - 输出：一个包含各类指标的字典（由 find_best_threshold + save_all_result 定义）
        """
        file_name = self.file_names[index]
        print(f'  [test_single] current file is {file_name}')
        self.file_now = file_name

        self.model.eval()

        all_scores = []
        test_labels = []
        test_datas = []

        # 按窗口点位的重构误差作为 anomaly score
        criterion = nn.MSELoss(reduction='none')
        lambda_js = getattr(self.args, "lambda_js", 0.01)

        # ===== Domain anchor (Phase1) =====
        gu_tre = torch.from_numpy(self.global_usage_tre).to(self.device)  # [Nt]
        gu_sea = torch.from_numpy(self.global_usage_sea).to(self.device)  # [Ns]
        gu_tre = gu_tre / (gu_tre.sum() + 1e-8)
        gu_sea = gu_sea / (gu_sea.sum() + 1e-8)

        def js_divergence(p, q, eps=1e-8):
            """
            p, q: [K], sum to 1
            """
            m = 0.5 * (p + q)
            kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
            kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
            return 0.5 * (kl_pm + kl_qm)

        for i, (batch_x, label_y) in enumerate(test_loader_single):
            batch_x = batch_x.float().to(self.device)

            # 前向传播
            X_hat, (Xt, Xs, Xr), (at, as_, ar), lambdas, alpha = self.model(batch_x)

            recon_score = criterion(X_hat, batch_x)

            # ---------- (2) Usage Drift ----------
            # window-level usage

            # Details will be released upon publication

            anomaly_score = recon_score + lambda_js * js_score

            anomaly_score = anomaly_score.detach().cpu().numpy()
            all_scores.append(anomaly_score)
            test_labels.append(label_y)

            if (i + 1) % 200 == 0:
                print("    [test_single] iters: {}, current batch mean score: {:.6f}".format(
                    i + 1, anomaly_score.mean()))

        # 拼接成全局向量
        all_scores = np.concatenate(all_scores, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        all_scores = np.array(all_scores)
        test_labels = np.array(test_labels)

        print("  [test_single] file: {} | total windows: {} | anomaly labels: {}".format(
            file_name, len(all_scores), int(test_labels.sum())))

        # 交给 find_best_threshold 做阈值搜索和指标计算
        gt = test_labels.astype(int)

        result = find_best_threshold(all_scores, gt)

        return None


