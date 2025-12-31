import os.path

from data_provider.data_loader import PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, SYNPatternSegLoader, \
    SYNGlobalSegLoader, SYNTrendSegLoader, SYNContextualSegLoader, NIPS_TS_WaterSegLoader, NIPS_TS_SwanSegLoader, \
    UCRSingleSegLoader, NewTSBADULoader
from torch.utils.data import DataLoader, ConcatDataset
import json

ucr_category = ['sddb', 'BIDM', 'MIS44AirTemperature', 'ECG', 'GP711MarkerLFM',
                'InternalBleeding', 'Lab2Cmac011215EPG', 'MesoplodonDensirostris',
                'PowerDemand', 'MARS', 'WalkingAceleration', 'apneaecg', 'gait',
                'insectEPG', 'ltstdbs3079', 'park3m', 'qtdbSel', 'resperation',
                's20101m', 'tiltAPB', 'CHARISfive', 'CHARISten', 'Italianpowerdemand',
                'STAFFIIIDatabase', 'Fantasia', 'longtermecg', 'taichidbS0715Master',
                'tilt127', 'weallwalk', 'EPG']
selected_TSB_MAD_classes = ['CATSv2', 'GHL', 'MITDB', 'MSL', 'SMAP', 'SMD', 'SVDB', 'TAO']

selected_TSB_U_classes = ['CATSv2', 'Daphnet', 'Exathlon', 'IOPS', 'LTDB', 'MGAB', 'MITDB', 'MSL', 'NAB', 'NEK',
                          'OPPORTUNITY', 'Power',
                          'SED', 'SMAP', 'SMD', 'SVDB', 'SWaT', 'Stock', 'TAO', 'TODS', 'UCR', 'WSD', 'YAHOO']
ignore_class_lists = [
                    # '531_SMAP_id_1_Sensor_tr_1811_1st_4510.csv', '534_SMAP_id_4_Sensor_tr_1968_1st_4370.csv',
                      '188_SMD_id_11_Facility_tr_500_1st_392.csv',
                      # '265_IOPS_id_6_WebService_tr_6453_1st_6553.csv'
                      ]
ignore_ucr_lists = ['069_UCR_Anomaly_DISTORTEDinsectEPG5_3200_8500_8501.txt', '177_UCR_Anomaly_insectEPG5_3200_8500_8501.txt']

def multi_data_provider(args, flag):
    '''

    :param args:
    :param flag:
    :return: file_list -> dict, datasets -> list
    '''

    class_now = args.file_class
    save_foloder = f'./data_provider/{args.data}/'


    if args.data == 'TSB_U':
        from data_provider.data_loader import NewTSBADULoader

        # if class_now not in selected_TSB_MAD_classes:
        #     raise NotImplementedError(f'{class_now} not implemented')

        file_list = os.listdir(args.root_path)
        file_list.sort()

        current_filenames = []
        current_dls = {}
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            if class_now in filename:
                current_filenames.append(filename)
                dataset_single = NewTSBADULoader(root_path=args.root_path, file_name=filename, file_class=class_now,
                                                 win_size=args.win_size, flag=flag)
                current_dls[filename] = dataset_single


    elif args.data == 'TSB_M':
        from data_provider.data_loader import NewTSBADMLoader
        file_list = os.listdir(args.root_path)
        file_list.sort()

        current_filenames = []
        current_dls = {}
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            if class_now in filename:
                current_filenames.append(filename)
                dataset_single = NewTSBADMLoader(root_path=args.root_path, file_name=filename, file_class=class_now,
                                                 win_size=args.win_size, kernel_size=args.kernel_size, flag=flag)
                current_dls[filename] = dataset_single


    elif args.data == 'UCR':
        from data_provider.data_loader import UCRSingleSegLoader
        if class_now not in ucr_category:
            raise NotImplementedError(f'{class_now} not implemented')

        # 读取当前分类的所有数据集
        file_list = os.listdir(args.root_path)
        file_list.sort()
        current_filenames = []
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            if class_now in filename:
                current_filenames.append(filename)

        current_dls = {}
        for filename in current_filenames:
            dataset_single = UCRSingleSegLoader(root_path=args.root_path, file_name=filename, file_class=class_now,
                                                 win_size=args.win_size, flag=flag)
            current_dls[filename] = dataset_single

    else:
        raise NotImplementedError(f'{args.data} not implemented')

    return current_filenames, current_dls







    #     if flag == 'test' or flag == 'thre' or flag == 'train_thre':
    #         # if detection, only need current data:
    #         for filename in current_filenames
    #
    #
    #         shuffle_flag = False
    #         batch_size = args.batch_size
    #         data_set = ConcatDataset(data_set)
    #         data_loader = [
    #             DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
    #             for ds in data_set.datasets
    #         ]
    #         filename = current_filenames
    #
    #
    #
    #
    #
    #     data_set = []  # 不同的数据集不同的dataset配合
    #
    #     if flag == 'test' or flag == 'thre' or flag == 'train_thre':
    #         shuffle_flag = False
    #         batch_size = args.batch_size  # bsz=1 for evaluation
    #         data_set = ConcatDataset(data_set)
    #         data_loader = [
    #             DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
    #             for ds in data_set.datasets
    #         ]
    #         filename = current_filenames
    #     else:
    #         shuffle_flag = True
    #         batch_size = args.batch_size  # bsz for train and valid
    #         drop_last = False
    #
    #         filename = current_filenames
    #         data_set = ConcatDataset(data_set)
    #         if args.is_multi_dataset:  # multi-datasets_together_train
    #             data_loader = DataLoader(
    #                 data_set,
    #                 batch_size=batch_size,
    #                 shuffle=shuffle_flag,
    #                 num_workers=args.num_workers,
    #                 drop_last=drop_last)
    #         else:
    #             data_loader = [
    #                 DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
    #                 for ds in data_set.datasets
    #             ]
    #
    #
    #
    # else:  # 如果只有一个数据集
    #     filename = [args.data]
    #     Data = data_dict[args.data]
    #     data_set = []
    #     data_set_single = Data(
    #         root_path=args.root_path,
    #         win_size=args.win_size,
    #         flag=flag,
    #     )
    #     data_set.append(data_set_single)
    #
    #     if flag == 'test' or flag == 'thre' or flag == 'train_thre':
    #         shuffle_flag = False
    #         batch_size = args.batch_size  # bsz=1 for evaluation
    #         data_set = ConcatDataset(data_set)
    #         data_loader = [
    #             DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
    #             for ds in data_set.datasets
    #         ]
    #     else:
    #         shuffle_flag = True
    #         batch_size = args.batch_size  # bsz for train and valid
    #         drop_last = False
    #
    #         data_set = ConcatDataset(data_set)
    #         if args.is_multi_dataset:  # multi-datasets_together_train
    #             data_loader = DataLoader(
    #                 data_set,
    #                 batch_size=batch_size,
    #                 shuffle=shuffle_flag,
    #                 num_workers=args.num_workers,
    #                 drop_last=drop_last)
    #         else:
    #             data_loader = [
    #                 DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag)
    #                 for ds in data_set.datasets
    #             ]
    #
    # return filename, data_loader
