import csv
import os

def save_all_evaluation_result(result, output_folder, file_now, args):
    pa_result, affi_result, ROC, vus_result = result
    (best_pa_threshold, pot_acc, pot_prec, pot_recall, pot_f1) = pa_result
    (best_affi_threshold, affi_P, affi_R, affi_f1) = affi_result
    (R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR) = vus_result['R_AUC_ROC'], vus_result['R_AUC_PR'], vus_result['VUS_ROC'], \
        vus_result['VUS_PR']

    file_name = os.path.join(output_folder, 'SOTA_MultiFusionAD_result_' + args.data + '.csv')
    file_exists = os.path.exists(file_name)

    if args.data == 'SMD_Multi':
        file_now = '\'' + file_now

    with open(file_name, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 如果文件不存在，添加头部行
        if not file_exists:
            writer.writerow(
                ['data', 'file', 'is_multi',
                 'is_select', 'is_prototype', 'is_sensitive', 'select_in_test'
                 'batch_size', 'win_size',
                 'ds_layer', 'ds_win', 'd_model',
                 'pot_pre', 'pot_recall', 'pot_f1',
                 'affi_pre', 'affi_recall', 'affi_f1',
                 'ROC', 'R_AUC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR'])
        # 写入数据行
        writer.writerow(
            [args.data, str(file_now), args.is_multi_dataset,
             args.select_auxset, args.proto_detect, args.sensitive_component, args.select_in_test,
             args.batch_size, args.win_size, args.down_sampling_layers,
             args.down_sampling_window, args.d_model,
             pot_prec, pot_recall, pot_f1,
             affi_P, affi_R, affi_f1,
             ROC, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR])
    pass




