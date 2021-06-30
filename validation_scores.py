"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score


def convert_beat_prediction(predictions, qrs_lists, beat_preQRS_length, beat_postQRS_length):
    beat_predictions = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        qrs_list = qrs_lists[i]
        beat_prediction = []

        for qrs in qrs_list:
            begin = qrs - beat_preQRS_length if qrs - beat_preQRS_length > 0 else 0
            end = qrs + beat_postQRS_length if qrs + beat_postQRS_length < len(prediction) else len(prediction)
            beat_prediction.append(np.max(prediction[begin:end]))

        if len(beat_prediction) > 0:
            beat_prediction = np.array(beat_prediction, dtype=np.float32)
            beat_predictions.append(beat_prediction)

    return beat_predictions


def convert_beat_annotation(annotations, qrs_lists, beat_preQRS_length, beat_postQRS_length):
    beat_annotations = []
    for i in range(len(annotations)):
        prediction = annotations[i]
        qrs_list = qrs_lists[i]
        beat_prediction = []

        for qrs in qrs_list:
            begin = qrs - beat_preQRS_length if qrs - beat_preQRS_length > 0 else 0
            end = qrs + beat_postQRS_length if qrs + beat_postQRS_length < len(prediction) else len(prediction)
            beat_prediction.append(np.round(np.mean(prediction[begin:end])))

        if len(beat_prediction) > 0:
            beat_prediction = np.array(beat_prediction, dtype=np.float32)
            beat_annotations.append(beat_prediction)

    return beat_annotations


def convert_beat_prediction2(predictions, qrs_lists, beat_preQRS_length, beat_postQRS_length):
    beat_predictions = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        qrs_list = qrs_lists[i]
        qrs_count = len(qrs_list)
        beat_prediction = []

        qrs_begin = qrs_list - beat_preQRS_length
        qrs_end = qrs_list + beat_postQRS_length

        qrs_begin[qrs_begin<0] = 0
        qrs_end[qrs_end>len(prediction)] = len(prediction)

        for j in range(qrs_count):
            begin = qrs_begin[j]
            end = qrs_end[j]
            if begin >= end:
                print('Prediction: {} record, {} beat, begin={}, end={}'.format(i,j,begin,end))
            beat_prediction.append(np.mean(prediction[begin:end], axis=0))

        if len(beat_prediction) > 0:
            beat_prediction = np.array(beat_prediction, dtype=np.float32)
            beat_predictions.append(beat_prediction)

    return beat_predictions


def convert_beat_annotation2(annotations, qrs_lists, beat_preQRS_length, beat_postQRS_length):
    beat_annotations = []
    for i in range(len(annotations)):
        annotation = annotations[i]
        qrs_list = qrs_lists[i]
        qrs_count = len(qrs_list)
        beat_prediction = np.zeros(qrs_count)

        qrs_begin = qrs_list - beat_preQRS_length
        qrs_end = qrs_list + beat_postQRS_length

        qrs_begin[qrs_begin<0] = 0
        qrs_end[qrs_end>len(annotation)] = len(annotation)

        for j in range(qrs_count):
            begin = qrs_begin[j]
            end = qrs_end[j]
            if begin >= end:
                print('Annotation: {} record, {} beat, begin={}, end={}'.format(i,j,begin,end))
            beat_prediction[j] = np.round(np.mean(annotation[begin:end]))

        if len(beat_prediction) > 0:
            beat_prediction = np.array(beat_prediction, dtype=np.float32)
            beat_annotations.append(beat_prediction)

    return beat_annotations


def get_roc_prc(predictions, annotations):

    AUROC, AUPRC = [], []

    for i in range(len(predictions)):

        prediction = predictions[i]
        annotation = annotations[i]

        if np.any(annotation) and len(np.unique(annotation)) == 2:
            precision, recall, thresholds = \
                precision_recall_curve(np.ravel(annotation),
                                       prediction,
                                       pos_label=1, sample_weight=None)
            auprc = auc(recall, precision)
            auroc = roc_auc_score(np.ravel(annotation), prediction)
            AUPRC.append(auprc)
            AUROC.append(auroc)
        else:
            auprc = auroc = float('nan')
        print(' AUROC:%f AUPRC:%f' % (auroc, auprc))

    AUROC = np.array(AUROC)
    AUPRC = np.array(AUPRC)
    print()
    print('Training AUROC Performance: %f+/-%f'
          % (np.mean(AUROC), np.std(AUROC)))
    print('Training AUPRC Performance: %f+/-%f'
          % (np.mean(AUPRC), np.std(AUPRC)))
    print()

    return AUROC, AUPRC


def get_tp_fp_fn(predictions, annotations, thres=0.5):

    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(predictions)):

        prediction = predictions[i]
        annotation = annotations[i]

        prediction_binary = prediction > thres
        annotation_binary = annotation > thres

        tp = np.count_nonzero(np.logical_and(prediction_binary, annotation_binary))
        fp = np.count_nonzero(np.logical_and(prediction_binary, np.logical_not(annotation_binary)))
        fn = np.count_nonzero(np.logical_and(np.logical_not(prediction_binary), annotation_binary))
        tn = np.count_nonzero(np.logical_and(np.logical_not(prediction_binary), np.logical_not(annotation_binary)))

        se = tp / (tp + fn + 1E-10)
        ppv = tp / (tp + fp + 1E-10)
        acc = (tp+tn) / (tp + fp + fn + tn + 1E-10)
        sp = tn / (tn + fp + 1E-10)

        print('{}: se = {}, sp = {}, ppv = {}, acc = {}'.format(i, se, sp, ppv, acc))

        TP += tp
        FP += fp
        FN += fn
        TN += tn

    Se = TP / (TP + FN + 1E-10)
    PPv = TP / (TP + FP + 1E-10)
    Acc = (TP + TN) / (TP + FP + FN + TN + 1E-10)
    Sp = TN / (TN + FP + 1E-10)

    print('TP = {}, FP = {}, FN = {}, TN= {}'.format(TP, FP, FN, TN))
    print('Se = {}, Sp = {}, PPv = {}, Acc = {}'.format(Se, Sp, PPv, Acc))

    return TP, FP, FN, Se, PPv, Acc, Sp


def get_se_ppv_acc_from_confmat(conf_mat):
    TP = conf_mat.diagonal()
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP
    TN = np.sum(conf_mat) - TP - FP - FN

    Se = TP / (TP + FN + 1E-10)
    Sp = TN / (TN + FP + 1E-10)
    PPv = TP / (TP + FP + 1E-10)
    Acc = (TP + TN) / (TP + FP + FN + TN + 1E-10)

    precisions = TP / (TP + FP + 1E-10)
    recalls = TP / (TP + FN + 1E-10)
    F1 = 2 * (precisions * recalls) / (precisions + recalls + 1E-10)

    return TP, FP, FN, Se, PPv, Acc, Sp, F1
