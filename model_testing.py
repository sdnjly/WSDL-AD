"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import glob
import os
from os import path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import validation_scores
from LSEaggregation import LSEaggregation
from MaskedGlobalMaxPooling1D import MaskedGlobalMaxPooling1D
from prepare_MITDB_data import load_file


def evaluate_model(dataset_test,
                   model,
                   model_characteristics,
                   model_folder='models',
                   beat_group_labels=None,
                   with_rrs=True,
                   with_qrs=False,
                   with_entropy=True,
                   timestampStr=None,
                   output_layer_name="predictions_beats"):

    # construct model
    if model is None:
        model_structure_file = path.join(model_folder, 'model_' + timestampStr + '.json')
        model_weights_file = path.join(model_folder, 'weights_' + timestampStr + '.model')

        custom_classes = {'LSEaggregation': LSEaggregation,
                          'MaskedGlobalMaxPooling1D': MaskedGlobalMaxPooling1D}

        with custom_object_scope(custom_classes):
            with open(model_structure_file, 'r') as f:
                model_json = f.read()

            model = model_from_json(model_json)
            # load weights of best model
            model.load_weights(model_weights_file).expect_partial()

    results_dataframe = None

    local_prediction_layer = model.get_layer(output_layer_name)
    ecg_layer = model.get_layer('ecg')
    inputs = [ecg_layer.output]
    if with_rrs:
        rrs_layer = model.get_layer('rrs')
        inputs += [rrs_layer.output]
    if with_entropy:
        rr_entropy_layer = model.get_layer('sample-entropy')
        inputs += [rr_entropy_layer.output]
    if with_qrs:
        qrs_layer = model.get_layer('qrs')
        inputs += [qrs_layer.output]

    mask_layer = model.get_layer('mask')
    inputs += [mask_layer.output]

    model_local = Model(inputs=inputs,
                        outputs=[local_prediction_layer.output])

    predictions = model_local.predict(dataset_test['X'])

    beat_predictions = validation_scores.convert_beat_prediction2(predictions, dataset_test['beats'],
                                                                  beat_preQRS_length=1,
                                                                  beat_postQRS_length=1)
    beat_annotations = dataset_test['beats_refs']

    beat_predictions = np.concatenate(beat_predictions, axis=0)
    beat_predictions = beat_predictions.argmax(axis=1).astype(np.float).squeeze()

    beat_annotations = np.concatenate(beat_annotations, axis=0)
    # beat_annotations = beat_annotations.argmax(axis=1).astype(np.float).squeeze()

    print('beat_predictions shape: {}'.format(beat_predictions.shape))
    print('beat_annotations shape: {}'.format(beat_annotations.shape))

    conf_mat = confusion_matrix(beat_annotations, beat_predictions)

    print(conf_mat)

    TP, FP, FN, Se, PPv, Acc, Sp, F1 = validation_scores.get_se_ppv_acc_from_confmat(conf_mat)

    results = {'id': timestampStr, 'output_layer': output_layer_name,
               'conf_mat': np.array2string(conf_mat),
               'dataset': dataset_test['dataset']}
    results = {**model_characteristics, **results}
    for i in range(len(TP)):
        print('{}: TP={}, FP={}, FN={}, Se={:.4f}, Sp={:.4f}, PPv={:.4f}, Acc={:.4f}, F1={:.4f}'.format(
            beat_group_labels[i], TP[i], FP[i], FN[i], Se[i], Sp[i], PPv[i], Acc[i], F1[i]))
        results[beat_group_labels[i] + '_Se'] = Se[i]
        results[beat_group_labels[i] + '_Ppr'] = PPv[i]
        results[beat_group_labels[i] + '_Acc'] = Acc[i]
        results[beat_group_labels[i] + '_F1'] = F1[i]
        results[beat_group_labels[i] + '_Sp'] = Sp[i]

    df = pd.DataFrame(data=results, index=[0])
    if results_dataframe is None:
        results_dataframe = df
    else:
        results_dataframe = results_dataframe.append(df)

    return results_dataframe


def test_wsl_model_from_file(input_paths,
                             model,
                             model_characteristics,
                             model_folder='models',
                             log_file=None,
                             files_list=None,
                             with_rrs=True,
                             with_qrs=False,
                             with_entropy=True,
                             timestampStr=None,
                             beat_labels=None,
                             beat_group_labels=None,
                             output_layer_name="predictions_beats",
                             seg_length=2 ** 12,
                             denoise=False,
                             rm_baseline=True,
                             normalization=True,
                             target_fs=250,
                             leads_names=None,
                             leads_number=1,
                             rrs_for_entropy_computing=20,
                             rrs_for_estimating_normal_rr=20,
                             rrs_normalize_for_entropy=True,
                             mm=1, r=0.05,
                             segment_margin=50,
                             relative_rr_scale_rate=5,
                             referring_annotation_type='beat',
                             annot_ext_name='atr',
                             show_record_wise_results=False,
                             save_PRC=False):

    # construct model
    if model is None:

        model_structure_file = path.join(model_folder, 'model_' + timestampStr + '.json')
        model_weights_file = path.join(model_folder, 'weights_' + timestampStr + '.model')
        custom_classes = {'LSEaggregation': LSEaggregation,
                          'MaskedGlobalMaxPooling1D': MaskedGlobalMaxPooling1D}

        with custom_object_scope(custom_classes):
            with open(model_structure_file, 'r') as f:
                model_json = f.read()

            model = model_from_json(model_json)
            # load weights of best model
            model.load_weights(model_weights_file).expect_partial()

    local_prediction_layer = model.get_layer(output_layer_name)
    ecg_layer = model.get_layer('ecg')
    inputs = [ecg_layer.output]
    if with_rrs:
        rrs_layer = model.get_layer('rrs')
        inputs += [rrs_layer.output]
    if with_entropy:
        rr_entropy_layer = model.get_layer('sample-entropy')
        inputs += [rr_entropy_layer.output]
    if with_qrs:
        qrs_layer = model.get_layer('qrs')
        inputs += [qrs_layer.output]

    mask_layer = model.get_layer('mask')
    inputs += [mask_layer.output]

    model_local = Model(inputs=inputs,
                        outputs=[local_prediction_layer.output])

    results_dataframe = None

    beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'S', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x']

    if not type(input_paths) == list:
        input_paths = [input_paths]
        files_list = [files_list]

    total_db_conf_mat = None
    total_record_conf_mat_list = []

    beat_labels_record = []
    beat_preds_record = []

    for path_id in range(len(input_paths)):

        db_beat_labels_record = []
        db_beat_preds_record = []

        input_path = input_paths[path_id]
        files = files_list[path_id]

        print("Testing on ", input_path)

        if os.path.basename(input_path) == 'afdb':
            invalid_files = ['00735', '03665']
            annot_qrs_ext_name = 'qrs'
        else:
            invalid_files = []
            annot_qrs_ext_name = 'atr'

        # get record list in the input path
        if files is None:
            files = glob.glob(os.path.join(input_path, '*.hea'))
            files.sort()
        else:
            for i in range(len(files)):
                files[i] = os.path.join(input_path, files[i])

        db_conf_mat = None

        record_conf_mat_list = []

        for file in files:
            filename = os.path.splitext(file)[0]
            basename = os.path.basename(filename)
            record_id = basename
            if basename in invalid_files:
                continue

            ecg_segs, qrs_map_segs, relative_rr_map_segs, rr_entropy_map_segs, \
            refs_segs, beatlist_segs, beatref_segs, beat_rawlabels_segs, \
            beat_rhythms_segs, record_id_segs = \
                load_file(filename, seg_length, 1,
                          denoise, rm_baseline, normalization,
                          target_fs, beat_labels, beat_group_labels,
                          beat_labels_all, leads_names, leads_number, rrs_for_entropy_computing,
                          rrs_for_estimating_normal_rr,
                          rrs_normalize_for_entropy,
                          mm, r, segment_margin, relative_rr_scale_rate,
                          referring_annotation_type, annot_ext_name,
                          annot_qrs_ext_name)

            ecg_segs = pad_sequences(ecg_segs, maxlen=seg_length, padding='post', truncating='post', dtype='float32')
            relative_rr_map_segs = pad_sequences(relative_rr_map_segs, maxlen=seg_length, padding='post',
                                                 truncating='post', dtype='float32')
            rr_entropy_map_segs = pad_sequences(rr_entropy_map_segs, maxlen=seg_length, padding='post',
                                                truncating='post', dtype='float32')
            qrs_map_segs = pad_sequences(qrs_map_segs, maxlen=seg_length, padding='post',
                                         truncating='post', dtype='float32')
            mask_segs = np.squeeze(qrs_map_segs, axis=-1)

            X = [ecg_segs]
            if with_rrs:
                X += [relative_rr_map_segs]
            if with_entropy:
                X += [rr_entropy_map_segs]
            if with_qrs:
                X += [qrs_map_segs]
            X += [mask_segs]

            predictions = model_local.predict(X)
            beat_predictions = validation_scores.convert_beat_prediction2(predictions, beatlist_segs,
                                                                          beat_preQRS_length=1,
                                                                          beat_postQRS_length=1)
            beat_annotations = beatref_segs

            beat_predictions = np.concatenate(beat_predictions, axis=0)
            beat_annotations = np.concatenate(beat_annotations, axis=0)

            db_beat_labels_record.append(beat_annotations)
            db_beat_preds_record.append(beat_predictions)

            beat_predictions = beat_predictions.argmax(axis=1).astype(np.float).squeeze()

            labels = [i for i in range(len(beat_group_labels))]
            record_conf_mat = confusion_matrix(beat_annotations, beat_predictions, labels=labels)

            record_conf_mat_list.append(record_conf_mat)
            total_record_conf_mat_list.append(record_conf_mat)

            if db_conf_mat is None:
                db_conf_mat = record_conf_mat
            else:
                db_conf_mat += record_conf_mat

            if show_record_wise_results:
                print(record_id)
                print(record_conf_mat)

        print(db_conf_mat)

        if total_db_conf_mat is None:
            total_db_conf_mat = db_conf_mat
        else:
            total_db_conf_mat += db_conf_mat

        beat_labels_record.extend(db_beat_labels_record)
        beat_preds_record.extend(db_beat_preds_record)

        class_counts = np.sum(db_conf_mat, axis=-1)

        TP, FP, FN, Se, PPv, Acc, Sp, F1 = validation_scores.get_se_ppv_acc_from_confmat(db_conf_mat)

        beat_labels_record_db = np.concatenate(db_beat_labels_record, axis=0)
        beat_labels_record_db = beat_labels_record_db.reshape(len(beat_labels_record_db), 1)
        label_encoder = OneHotEncoder(sparse=False)
        beat_labels_record_db = label_encoder.fit_transform(beat_labels_record_db)
        beat_preds_record_db = np.concatenate(db_beat_preds_record, axis=0)

        # compute PRC
        precision_list = []
        recall_list = []
        for i in range(beat_preds_record_db.shape[1]):
            precision, recall, _ = precision_recall_curve(beat_labels_record_db[:, i], beat_preds_record_db[:, i])
            precision_list.append(precision)
            recall_list.append(recall)

        AUPRC = average_precision_score(beat_labels_record_db, beat_preds_record_db, average=None)

        results = {'id': timestampStr, 'output_layer': output_layer_name,
                   'conf_mat': np.array2string(db_conf_mat),
                   'dataset': input_path}
        results = {**model_characteristics, **results}
        for i in range(len(TP)):
            print('{}: Counts={}, TP={}, FP={}, FN={}, Se={:.4f}, Sp={:.4f}, PPv={:.4f}, Acc={:.4f}, F1={:.4f}, AUPRC={:.4f}'.format(
                beat_group_labels[i], class_counts[i], TP[i], FP[i], FN[i], Se[i], Sp[i], PPv[i], Acc[i], F1[i], AUPRC[i]))
            results[beat_group_labels[i] + '_Se'] = Se[i]
            results[beat_group_labels[i] + '_Ppr'] = PPv[i]
            results[beat_group_labels[i] + '_Acc'] = Acc[i]
            results[beat_group_labels[i] + '_F1'] = F1[i]
            results[beat_group_labels[i] + '_Sp'] = Sp[i]
            results[beat_group_labels[i] + '_AUPRC'] = AUPRC[i]

        print(results)
        df = pd.DataFrame(data=results, index=[0])

        if log_file is not None:
            df.to_csv(log_file, mode='a')

    beat_labels_record_all = np.concatenate(beat_labels_record, axis=0)
    beat_labels_record_all = beat_labels_record_all.reshape(len(beat_labels_record_all), 1)
    label_encoder = OneHotEncoder(sparse=False)
    beat_labels_record_all = label_encoder.fit_transform(beat_labels_record_all)
    beat_preds_record_all = np.concatenate(beat_preds_record, axis=0)

    # compute PRC
    precision_list = []
    recall_list = []
    for i in range(beat_preds_record_all.shape[1]):
        precision, recall, _ = precision_recall_curve(beat_labels_record_all[:, i], beat_preds_record_all[:, i])
        precision_list.append(precision)
        recall_list.append(recall)

    AUPRC = average_precision_score(beat_labels_record_all, beat_preds_record_all, average=None)
    PRC = {'precision_list':precision_list, 'recall_list': recall_list}
    if save_PRC:
        joblib.dump(PRC, 'models/PRC_' + timestampStr + '.sav', protocol=0)

    print('Total metrics of all the test DBs.')
    TP, FP, FN, Se, PPv, Acc, Sp, F1 = validation_scores.get_se_ppv_acc_from_confmat(total_db_conf_mat)
    results = {'id': timestampStr, 'output_layer': output_layer_name,
               'conf_mat': np.array2string(total_db_conf_mat),
               'dataset': 'total_db'}
    results = {**model_characteristics, **results}
    class_counts = np.sum(total_db_conf_mat, axis=-1)
    for i in range(len(TP)):
        print('{}: Counts={}, TP={}, FP={}, FN={}, Se={:.4f}, Sp={:.4f}, PPv={:.4f}, Acc={:.4f}, F1={:.4f}, AUPRC={:.4f}'.format(
            beat_group_labels[i], class_counts[i], TP[i], FP[i], FN[i], Se[i], Sp[i], PPv[i], Acc[i], F1[i], AUPRC[i]))
        results[beat_group_labels[i] + '_Se'] = Se[i]
        results[beat_group_labels[i] + '_Ppr'] = PPv[i]
        results[beat_group_labels[i] + '_Acc'] = Acc[i]
        results[beat_group_labels[i] + '_F1'] = F1[i]
        results[beat_group_labels[i] + '_Sp'] = Sp[i]
        results[beat_group_labels[i] + '_AUPRC'] = AUPRC[i]

    df = pd.DataFrame(data=results, index=[0])
    if log_file is not None:
        df.to_csv(log_file, mode='a')

    return results_dataframe
