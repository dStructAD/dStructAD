import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events
from utils.vus.metrics import get_range_vus_roc



def point_adjust(pred, gt):
    pred = pred.astype(int)
    gt = gt.astype(int)

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')

    return accuracy, precision, recall, f_score


def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P, R, F


def find_best_threshold(score, true_label_all):
    loss_lb, loss_ub = np.quantile(score, 0.01), np.quantile(score, 0.999)
    print(f'min_loss: {loss_lb}, max_loss: {loss_ub}')

    pa_f1_threshold_all = {}
    affi_f1_threshold_all = {}
    for loss_threshold in np.linspace(loss_lb, loss_ub, 100):
        pred_label = (score > loss_threshold).astype(int)

        # point adjust
        try:
            pot_acc, pot_prec, pot_recall, pot_f1 = point_adjust(pred_label, true_label_all)
        except:
            pot_acc, pot_prec, pot_recall, pot_f1 = 0, 0, 0, 0
        pa_f1_threshold_all[loss_threshold] = pot_f1

        # affi f1
        try:
            affi_P, affi_R, affi_f1 = getAffiliationMetrics(pred_label, true_label_all)
        except:
            affi_P, affi_R, affi_f1 = 0, 0, 0
        affi_f1_threshold_all[loss_threshold] = affi_f1
    vus_results = get_range_vus_roc(score, true_label_all, 100)  # default slidingWindow = 100
    print(vus_results)

    ori_auc = roc_auc_score(true_label_all, score)
    print("ori_auc: {:0.4f}".format(ori_auc))

    best_pa_threshold = max(pa_f1_threshold_all, key=pa_f1_threshold_all.get)
    pred_label = (score > best_pa_threshold).astype(int)
    pot_acc, pot_prec, pot_recall, pot_f1 = point_adjust(pred_label, true_label_all)

    print(f'pa threshold is {best_pa_threshold}:')
    print("Point Adjust Accuracy : {:0.4f}, Pot-Precision : {:0.4f}, Pot-Recall : {:0.4f}, Pot-F-score : {:0.4f} ".format(
            pot_acc, pot_prec, pot_recall, pot_f1))

    best_affi_threshold = max(affi_f1_threshold_all, key=affi_f1_threshold_all.get)
    pred_label = (score > best_pa_threshold).astype(int)
    affi_P, affi_R, affi_f1 = getAffiliationMetrics(true_label_all, pred_label)
    print(f'affi threshold is {best_affi_threshold}:')
    print("AffiPrecision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f} ".format(
        affi_P, affi_R, affi_f1))


    return (best_pa_threshold, pot_acc, pot_prec, pot_recall, pot_f1), (best_affi_threshold, affi_P, affi_R, affi_f1), ori_auc, vus_results
