import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn

import warnings

# from model.model_utils import series_decomp

warnings.filterwarnings('ignore')


UCR_decom_win_sizes = {
    'ECG': {0: 101},
    'BIDM': {0: 201},
    'sddb': {0: 401},
}

MSL_decom_win_sizes = {0: 101,
  1: 101,
  2: 101,
  3: 101,
  4: 101,
  5: 101,
  6: 101,
  7: 101,
  8: 101,
  9: 101,
  10: 101,
  11: 101,
  12: 101,
  13: 101,
  14: 101,
  15: 101,
  16: 101,
  17: 101,
  18: 101,
  19: 101,
  20: 101,
  21: 101,
  22: 101,
  23: 101,
  24: 101,
  25: 101,
  26: 101,
  27: 101,
  28: 101,
  29: 101,
  30: 101,
  31: 101,
  32: 101,
  33: 101,
  34: 101,
  35: 101,
  36: 101,
  37: 101,
  38: 101,
  39: 101,
  40: 101,
  41: 101,
  42: 101,
  43: 101,
  44: 101,
  45: 101,
  46: 101,
  47: 101,
  48: 101,
  49: 101,
  50: 101,
  51: 101,
  52: 101,
  53: 101,
  54: 101}

SMD_decom_win_sizes={0: 101,
  1: 101,
  2: 101,
  3: 101,
  4: 101,
  5: 101,
  6: 101,
  7: 101,
  8: 101,
  9: 101,
  10: 101,
  11: 101,
  12: 101,
  13: 101,
  14: 101,
  15: 101,
  16: 101,
  17: 101,
  18: 101,
  19: 101,
  20: 101,
  21: 101,
  22: 101,
  23: 101,
  24: 101,
  25: 101,
  26: 101,
  27: 101,
  28: 101,
  29: 101,
  30: 101,
  31: 101,
  32: 101,
  33: 101,
  34: 101,
  35: 101,
  36: 101,
  37: 101}

SWaT_decom_win_sizes={0: 3905,
 1: 31789,
 2: 1751,
 3: 1517,
 4: 101,
 5: 345,
 6: 119,
 7: 13519,
 8: 3793,
 9: 1517,
 10: 101,
 11: 101,
 12: 1517,
 13: 101,
 14: 1517,
 15: 101,
 16: 32197,
 17: 3573,
 18: 19941,
 19: 101,
 20: 8049,
 21: 8049,
 22: 1751,
 23: 101,
 24: 2753,
 25: 9319,
 26: 21439,
 27: 101,
 28: 19747,
 29: 101,
 30: 101,
 31: 101,
 32: 101,
 33: 101,
 34: 101,
 35: 16079,
 36: 8049,
 37: 351,
 38: 101,
 39: 101,
 40: 101,
 41: 101,
 42: 101,
 43: 101,
 44: 1399,
 45: 101,
 46: 813,
 47: 8049,
 48: 101,
 49: 101,
 50: 101}

maximum_decom_win_size = 901

def decom_raw_series(x_input, kernel_size):
    x_input = torch.from_numpy(x_input)

    # 先获取Trend信息
    front = x_input[0].repeat((kernel_size - 1) // 2)
    end = x_input[-1].repeat((kernel_size - 1) // 2)
    x_input_rolling = torch.cat([front, x_input, end])

    # 使用nn.Avg进行滑动平均，得到Trend
    avg_fun = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    x_trend = avg_fun(x_input_rolling.unsqueeze(0)).squeeze()

    x_season = x_input - x_trend

    return x_trend.numpy(), x_season.numpy()



class MyDatasets(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.root_path = root_path
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = None
        self.train = None
        self.train_trend = None
        self.test = None
        self.test_trend = None
        self.val = None
        self.val_trend = None
        self.test_labels = None

    # def __len__(self):
    #     """
    #     Number of images in the object data.
    #     """
    #     if self.flag == "train":
    #         return (self.train.shape[0] - self.win_size) // self.step + 1
    #     elif (self.flag == 'val'):
    #         return (self.val.shape[0] - self.win_size) // self.step + 1
    #     elif (self.flag == 'test'):
    #         return (self.test.shape[0] - self.win_size) // self.step + 1
    #     elif self.flag == 'train_thre':
    #         return (self.train.shape[0] - self.win_size) // self.win_size + 1
    #     else:
    #         return (self.test.shape[0] - self.win_size) // self.win_size + 1
    #

    def __len__(self): # 这个版本是将每个序列看做一个点
        """
        Number of images in the object data.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.flag == 'train_thre':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.flag == 'test_window':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return NotImplemented(f'{self.flag} is not implemented')

    def ST_ratio(self, data, data_trend):
        data_season = data - data_trend
        ss_data = np.sum(data ** 2, axis=0)
        ss_trend = np.sum(data_trend ** 2, axis=0)
        ss_season = np.sum(data_season ** 2, axis=0)

        self.trend_ratio = ss_trend / (ss_data + 1e-6)
        self.season_ratio = ss_season / (ss_data + 1e-6)



    def __getitem__(self, index_in): #
        train_s = index_in * self.step
        train_e = train_s + self.win_size

        test_s = index_in * self.step
        test_e = test_s + self.win_size

        if self.flag == "train":
            return self.train[train_s:train_e]
        elif self.flag == "val":
            return self.val[train_s:train_e]
        elif self.flag == 'test':
            test_label = self.test_labels[train_s:train_e]
            # test_label = (test_label.sum() > (self.win_size / 6)).astype(int)
            return self.test[train_s:train_e], test_label
        elif self.flag == 'train_thre':
            return self.train[test_s:test_e]
        elif self.flag == 'test_window':
            start = index_in // self.step * self.win_size
            end = index_in // self.step * self.win_size+ self.win_size
            return self.test[start:end], self.test_labels[start:end]
        else:
            raise NotImplementedError


class PSMSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)


class NewTSBADULoader(MyDatasets):
    def __init__(self, root_path, file_name, file_class, win_size, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scale = scale

        vals = file_name.split('.')[0].split('_')
        train_end = int(vals[6])

        df = pd.read_csv(os.path.join(root_path, file_name)).dropna()
        temp = df.iloc[:, 0:-1].values.astype(float)

        test_data_label = df['Label'].astype(int).to_numpy()[train_end:]
        train_data, test_data = temp[:train_end], temp[train_end:]

        if self.scale and 'SMD' not in file_name:
            if 'SMAP' in file_name:
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(temp)
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)


        self.train = train_data.reshape(-1)


        train_vali_split = 0.3


        self.val = self.train[(int)(len(self.train) * train_vali_split):]

        self.test = test_data.reshape(-1)
        self.test_labels = test_data_label



class NewTSBADMLoader(MyDatasets):
    def __init__(self, root_path, file_name, file_class, win_size, kernel_size=101, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scale = scale

        vals = file_name.split('.')[0].split('_')
        train_end = int(vals[6])

        df = pd.read_csv(os.path.join(root_path, file_name)).dropna()
        temp = df.iloc[:, 0:-1].values.astype(float)

        test_data_label = df['Label'].astype(int).to_numpy()[train_end:]
        train_data, test_data = temp[:train_end], temp[train_end:]

        if self.scale:

            self.scaler.fit(train_data)
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)


        data_trend, test_data_trend = train_data.copy(), test_data.copy()
        for col_num in range(train_data.shape[1]):
            data_trend[:, col_num], _ = decom_raw_series(train_data[:, col_num], kernel_size)
            test_data_trend[:, col_num], _ = decom_raw_series(test_data[:, col_num], kernel_size)

        self.ST_ratio(train_data, data_trend)

        self.train = train_data
        self.train_trend = data_trend


        train_vali_split = 0.8
        if file_class == 'MSL' or file_class == 'SMD' or file_class == 'SVDB' or file_class == 'TAO':
            self.val = test_data
            self.val_trend = test_data_trend
        else:
            self.val = self.train[(int)(len(self.train) * train_vali_split):]
            self.val_trend = self.train_trend[(int)(len(self.train) * train_vali_split):]

        self.test = test_data
        self.test_labels = test_data_label



class MSLSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)



class SMAPSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        cols = [0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 15, 16, 19, 20]
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        # data = data[:, cols]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        # test_data = test_data[:, cols]
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)



class SMDSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))



# from data_provider.swat_win_size import swat_trend_win_size
class SWATSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scale = scale


        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[12000:, :-1]
        test_data = test_data.values[:, :-1]


        if self.scale:

            self.scaler.fit(train_data)

            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

        data_trend, test_data_trend = train_data.copy(), test_data.copy()
        for col_num in range(train_data.shape[1]):
            if SWaT_decom_win_sizes[col_num] > maximum_decom_win_size:
                kernel_size = maximum_decom_win_size
            else:
                kernel_size = SWaT_decom_win_sizes[col_num]
            data_trend[:, col_num], _ = decom_raw_series(train_data[:, col_num], kernel_size)
            test_data_trend[:, col_num], _ = decom_raw_series(test_data[:, col_num], kernel_size)

        self.ST_ratio(train_data, data_trend)

        self.train = train_data
        self.train_trend = data_trend


        train_vali_split = 0.8
        self.val = self.train[(int)(len(self.train) * train_vali_split):]
        self.val_trend = self.train_trend[(int)(len(self.train) * train_vali_split):]

        self.test = test_data
        self.test_trend = test_data_trend
        self.test_labels = labels

        # self.scaler.fit(train_data)
        # train_data = self.scaler.transform(train_data)
        # test_data = self.scaler.transform(test_data)
        # self.train = train_data
        # self.test = test_data
        # self.val = train_data[int(0.8*len(train_data)):]
        # self.test_labels = labels
        # if flag == "train" or flag == "train_thre":
        #     self.train_trend = self.decompose(self.train)
        # elif (self.flag == 'val'):
        #     self.train_trend = self.decompose(self.train)
        #     self.val_trend = self.train_trend[int(0.8*len(train_data)):]
        # else:
        #     self.test_trend = self.decompose(self.test)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def decompose(self, data):
        trend_all = []
        for i in range(data.shape[1]):
            trend_now, season_now = decom_raw_series(data[:, i], 501)
            trend_all.append(trend_now.numpy())
        trend_all = np.array(trend_all).T
        return trend_all




class NIPS_TS_WaterSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(root_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(root_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(root_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)


class NIPS_TS_SwanSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(root_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(root_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(root_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)




class UCRSingleSegLoader(MyDatasets):
    def __init__(self, root_path, file_name, file_class, win_size, kernel_size=101, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scale = scale

        vals = file_name.split('.')[0].split('_')
        dnum, vals = int(vals[0]), vals[-3:]
        vals = [int(i) for i in vals]
        temp = np.genfromtxt(os.path.join(root_path, file_name))
        temp = np.array(temp, dtype=np.float32)
        data, test_data = temp[:vals[0]], temp[vals[0]:]
        labels_now = np.zeros_like(test_data)
        labels_now[vals[1] - vals[0]:vals[2] - vals[0]] = 1


        if self.scale:
            data = data.reshape(-1, 1)
            test_data = test_data.reshape(-1, 1)
            self.scaler.fit(data)
            data = self.scaler.transform(data)
            test_data = self.scaler.transform(test_data)

            data = data.reshape(-1)
            test_data = test_data.reshape(-1)


        # data = data.reshape(-1, 1)
        # data_trend = data.copy()
        # test_data = test_data.reshape(-1, 1)
        # test_data_trend = test_data.copy()


        self.train = data

        train_vali_split = 0.8
        val_start = int(train_vali_split*len(data))
        if len(data[val_start:]) <= win_size:
            val_start = int(0.1*len(data))

        self.val = data[val_start:]

        self.test = test_data
        self.test_labels = labels_now
        print("test:", self.test.shape)
        print("train:", self.train.shape)







class SYNPatternSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'normal_one_dim.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv'))
        test_data = test_data.values[:, :1]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = data[int(0.5*len(data)):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)


class SYNGlobalSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(os.path.join(root_path, 'normal_one_dim.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'global_one_dim.csv'))
        test_data = test_data.values[:, :1]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = data[int(0.25*len(data)):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'global_one_dim.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)



class SYNTrendSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'normal_one_dim.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv'))
        test_data = test_data.values[:, :1]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = data[int(0.5*len(data)):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)



class SYNContextualSegLoader(MyDatasets):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'normal_one_dim.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv'))
        test_data = test_data.values[:, :1]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = data[int(0.5*len(data)):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'pattern_one_dim.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)



class SMDSingleLoader(MyDatasets):
    def __init__(self, root_path, filename, win_size, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scale = scale
        self.scaler = StandardScaler()
        self.root_path = root_path
        self.file_name = filename
        self.__read_data__()

    def __read_data__(self):

        train_file_path = os.path.join(self.root_path, 'train', 'machine-' + self.file_name + '.txt')
        test_file_path = os.path.join(self.root_path, 'test', 'machine-' + self.file_name + '.txt')
        test_label_file_path = os.path.join(self.root_path, 'test_label', 'machine-' + self.file_name + '.txt')

        train_data = np.loadtxt(train_file_path, delimiter=',')
        test_data = np.loadtxt(test_file_path, delimiter=',')
        test_data_label = np.loadtxt(test_label_file_path, delimiter=',')

        if self.scale:

            self.scaler.fit(train_data)

            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

        data_trend, test_data_trend = train_data.copy(), test_data.copy()
        for col_num in range(train_data.shape[1]):
            kernel_size = SMD_decom_win_sizes[col_num]
            data_trend[:, col_num], _ = decom_raw_series(train_data[:, col_num], kernel_size)
            test_data_trend[:, col_num], _ = decom_raw_series(test_data[:, col_num], kernel_size)

        self.ST_ratio(train_data, data_trend)

        self.train = train_data
        self.train_trend = data_trend


        train_vali_split = 0.8
        self.val = self.train[(int)(len(self.train) * train_vali_split):]
        self.val_trend = self.train_trend[(int)(len(self.train) * train_vali_split):]

        self.test = test_data
        self.test_trend = test_data_trend
        self.test_labels = test_data_label





class MSLSingleLoader(MyDatasets):
    def __init__(self, root_path, filename, win_size, step=1, flag="train", scale=True):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scale = scale
        self.scaler = StandardScaler()
        self.root_path = root_path
        self.file_name = filename
        self.__read_data__()

    def __read_data__(self):
        import ast

        train_file_path = os.path.join(self.root_path, 'data', 'train', self.file_name + '.npy')
        test_file_path = os.path.join(self.root_path, 'data', 'test', self.file_name + '.npy')


        train_data = np.load(train_file_path)
        test_data = np.load(test_file_path)


        # 读取label
        test_label_path =os.path.join(self.root_path, 'labeled_anomalies.csv')
        labeled_anomalies_all = pd.read_csv(test_label_path)
        MSL_list = labeled_anomalies_all['spacecraft'] == 'MSL'
        labeled_anomalies_msl = labeled_anomalies_all[MSL_list].set_index('chan_id')
        labels = labeled_anomalies_msl.loc[self.file_name, 'anomaly_sequences']
        label_type = labeled_anomalies_msl.loc[self.file_name, 'class']

        labels = ast.literal_eval(labels)
        label_type = np.array(label_type.strip('[]').replace(" ", "").split(','))

        labels_all = np.zeros(len(test_data))
        for label_now, class_now in zip(labels, label_type):
            if class_now == 'contextual':
                labels_all[label_now[0]: label_now[1]] = 1
            elif class_now == 'point':
                labels_all[label_now[0]: label_now[1]] = 1
            else:
                raise NotImplementedError(f'{class_now} not imple')

        # 数据scaler
        test_data_label = labels_all

        if self.scale:

            self.scaler.fit(train_data)

            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

        data_trend, test_data_trend = train_data.copy(), test_data.copy()
        for col_num in range(train_data.shape[1]):
            kernel_size = MSL_decom_win_sizes[col_num]
            data_trend[:, col_num], _ = decom_raw_series(train_data[:, col_num], kernel_size)
            test_data_trend[:, col_num], _ = decom_raw_series(test_data[:, col_num], kernel_size)

        self.ST_ratio(train_data, data_trend)

        self.train = train_data
        self.train_trend = data_trend

        train_vali_split = 0.8
        self.val = self.train[(int)(len(self.train) * train_vali_split):]
        self.val_trend = self.train_trend[(int)(len(self.train) * train_vali_split):]

        self.test = test_data
        self.test_trend = test_data_trend
        self.test_labels = test_data_label


        # self.train = train_data
        # self.test = test_data
        # self.test_labels = test_data_label
        #
        # self.val = self.train[(int)(len(self.train) * 0.8):]
        # self.val = self.test

        # print(self.file_name, len(self.train), len(self.val), sum(self.test_labels))