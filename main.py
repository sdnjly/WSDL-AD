"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import joblib
import numpy as np
import argparse

import data_utils
import prepare_MITDB_data
from model_testing import test_wsl_model_from_file, evaluate_model
from model_training import train_wsl_model
from prepare_CINC_data import load_data_simple

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_training_data_for_wsl(training_path, validation_path,
                               signal_length,
                               validation_files=None,
                               invalid_validation_files=None,
                               with_rrs=True,
                               with_entropy=True,
                               beat_labels=None,
                               beat_group_labels=None,
                               snomed_ct_classes=None,
                               equivalent_classes=None,
                               training_leads=None,
                               validation_leads_names=None,
                               validation_leads_number=1,
                               denoise=False,
                               normalization=True,
                               class_sample_weights=None,
                               relative_rr_scale_rate=5,
                               rrs_for_estimating_normal_rr=100,
                               rrs_for_entropy_computing = 100,
                               persistent_label_index=None,
                               excluded_classes=None,
                               referring_annotation_type='beat',
                               annot_qrs_ext_name='atr',
                               target_fs=250,
                               used_db_ids=None,
                               init_data_proportion=0,
                               validate_on_whole=False):
    # load data
    cinc_paths = [
        os.path.join(training_path, 'CPSC/Training_WFDB'),
        os.path.join(training_path, 'CPSC2/Training_2'),
        os.path.join(training_path, 'PTB-XL/WFDB'),
        os.path.join(training_path, 'E/WFDB'),
        os.path.join(training_path, 'WFDB_ShaoxingUniv'),
    ]

    if used_db_ids is not None:
        cinc_paths = [cinc_paths[i] for i in used_db_ids]

    signal_length_eval = signal_length

    X = []
    Y = []
    X_qrs_map = []
    X_relative_RR_feature_map = []
    X_RR_entropy = []
    QRSs = []
    file_names = []
    raw_labels = []

    if training_leads is None:
        training_leads = [1]

    for cinc_path in cinc_paths:
        print('loading training data: ', cinc_path)
        X_, Y_, X_qrs_map_, X_relative_RR_feature_map_, X_RR_entropy_, QRSs_, file_names_, raw_labels_, lengths = \
            load_data_simple(cinc_path, header_files=None, target_length=signal_length,
                             fs=target_fs, leads=training_leads, classes=snomed_ct_classes,
                             necessary_classes=[],
                             excluded_classes=excluded_classes,
                             resample_signal=True, remove_bw=True,
                             denoise=denoise, normalization=normalization,
                             normal_selecting_rate=1, equivalent_classes=equivalent_classes,
                             keep_ratio_of_N=1,
                             rrs_for_estimating_normal_rr=rrs_for_estimating_normal_rr,
                             relative_rr_scale_rate=relative_rr_scale_rate,
                             rrs_for_entropy_computing=rrs_for_entropy_computing,
                             rrs_normalize_for_entropy=True,
                             mm=1, r=0.05,
                             persistent_label_index=persistent_label_index)

        print('total records: ', len(X_))
        print('normal numbers: ', (Y_[:, 1:].sum(axis=1) == 0).astype(np.int).sum())
        print('label distribution: ')
        for i in range(len(beat_group_labels)):
            print(beat_group_labels[i] + ': ' + str(Y_[:, i].sum()))

        print('min length: ', min(lengths))
        print('max length: ', max(lengths))

        if len(X) == 0:
            X, Y, X_qrs_map, X_relative_RR_feature_map, X_RR_entropy, QRSs, file_names, raw_labels = \
                X_, Y_, X_qrs_map_, X_relative_RR_feature_map_, X_RR_entropy_, QRSs_, file_names_, raw_labels_
        else:
            X = np.concatenate([X, X_], axis=0)
            Y = np.concatenate([Y, Y_], axis=0)
            X_qrs_map = np.concatenate([X_qrs_map, X_qrs_map_], axis=0)
            X_relative_RR_feature_map = np.concatenate([X_relative_RR_feature_map, X_relative_RR_feature_map_], axis=0)
            X_RR_entropy = np.concatenate([X_RR_entropy, X_RR_entropy_], axis=0)
            QRSs.extend(QRSs_)
            file_names.extend(file_names_)
            raw_labels.extend(raw_labels_)

    mask_train = np.squeeze(X_qrs_map, axis=-1)

    print('loading validation data.')

    X_init, Y_init, qrs_positions_init, preRR_init, rr_entropy_seq_init, beats_init, beats_refs_init, beat_rawlabels_init, beat_rhythms_init, beat_record_id_init, \
    X_eval, Y_eval, qrs_positions_eval, preRR_eval, rr_entropy_seq_eval, beats_eval, beats_refs_eval, beat_rawlabels_eval, beat_rhythms_eval, beat_record_id_eval, \
        = \
        prepare_MITDB_data.load_data(validation_path,
                                     seg_length=signal_length_eval,
                                     files=validation_files,
                                     invalid_files=invalid_validation_files,
                                     target_fs=target_fs,
                                     train_proportion=init_data_proportion,
                                     denoise=denoise,
                                     rm_baseline=True,
                                     normalization=normalization,
                                     beat_labels=beat_labels,
                                     beat_group_labels=beat_group_labels,
                                     leads_names=validation_leads_names,
                                     leads_number=validation_leads_number,
                                     rrs_for_entropy_computing=rrs_for_entropy_computing,
                                     rrs_for_estimating_normal_rr=rrs_for_estimating_normal_rr,
                                     rrs_normalize_for_entropy=True,
                                     mm=1, r=0.05,
                                     segment_margin=50,
                                     relative_rr_scale_rate=relative_rr_scale_rate,
                                     referring_annotation_type=referring_annotation_type,
                                     annot_qrs_ext_name=annot_qrs_ext_name)

    # merge init into eval
    if validate_on_whole and len(X_init) > 0:
        X_eval = np.concatenate([X_eval, X_init], axis=0)
        Y_eval = np.concatenate([Y_eval, Y_init], axis=0)
        qrs_positions_eval = np.concatenate([qrs_positions_eval, qrs_positions_init], axis=0)
        preRR_eval = np.concatenate([preRR_eval, preRR_init], axis=0)
        rr_entropy_seq_eval = np.concatenate([rr_entropy_seq_eval, rr_entropy_seq_init], axis=0)
        beats_eval = beats_eval + beats_init
        beats_refs_eval = beats_refs_eval + beats_refs_init

    mask_eval = np.squeeze(qrs_positions_eval, axis=-1)
    beatref_sequence_eval = data_utils.compute_beatref_sequence(signal_length, beats_eval,
                                                                beats_refs_eval, len(beat_group_labels))
    if init_data_proportion > 0:
        mask_init = np.squeeze(qrs_positions_init, axis=-1)
        beatref_sequence_init = data_utils.compute_beatref_sequence(signal_length, beats_init,
                                                                    beats_refs_init, len(beat_group_labels))
    else:
        mask_init = None
        beatref_sequence_init = None

    inputs_train = [X]
    inputs_eval = [X_eval]
    inputs_init = [X_init]

    if with_rrs:
        inputs_train += [X_relative_RR_feature_map]
        inputs_eval += [preRR_eval]
        inputs_init += [preRR_init]

    if with_entropy:
        inputs_train += [X_RR_entropy]
        inputs_eval += [rr_entropy_seq_eval]
        inputs_init += [rr_entropy_seq_init]

    inputs_train += [mask_train]
    inputs_eval += [mask_eval]
    inputs_init += [mask_init]

    if class_sample_weights is not None:
        class_sample_weights = np.array(class_sample_weights, dtype=np.float32)
        class_sample_weights = np.expand_dims(class_sample_weights, axis=-1)

        sample_weights = np.matmul(Y, class_sample_weights)
        base_weight = np.amin(class_sample_weights)
        sample_weights[sample_weights < base_weight] = base_weight
    else:
        sample_weights = np.ones_like(Y[:, 0], dtype=np.float32)

    print('label distribution: ')
    for i in range(len(beat_group_labels)):
        print(beat_group_labels[i] + ': ' + str(Y[:, i].sum()))

    dataset_train = {'X': inputs_train,
                     'Y': Y}
    dataset_eval = {'X': inputs_eval,
                    'Y': Y_eval,
                    'Y_local': beatref_sequence_eval,
                    'beats': beats_eval,
                    'beats_refs': beats_refs_eval,
                    'dataset': 'validation'}
    dataset_init = {'X': inputs_init,
                    'Y': Y_init,
                    'Y_local': beatref_sequence_init,
                    'beats': beats_init,
                    'beats_refs': beats_refs_init,
                    'dataset': 'init-db'}

    return dataset_train, dataset_eval, dataset_init, sample_weights


def load_training_data_for_sl(training_set_path,
                              signal_length,
                              files=None,
                              invalid_files=None,
                              train_proportion=0.8,
                              with_rrs=True,
                              with_entropy=True,
                              beat_labels=None,
                              beat_group_labels=None,
                              test_leads_names=None,
                              test_leads_number=1,
                              denoise=False,
                              normalization=True,
                              class_sample_weights=None,
                              relative_rr_scale_rate=5,
                              rrs_for_estimating_normal_rr=100,
                              rrs_for_entropy_computing=100,
                              referring_annotation_type='beat',
                              annot_qrs_ext_name='atr',
                              target_fs=250):

    X_train, Y_train, qrs_positions_train, preRR_train, rr_entropy_seq_train, beats_train, beats_refs_train, beat_rawlabels_train, beat_rhythms_train, beat_record_id_train, \
    X_eval, Y_eval, qrs_positions_eval, preRR_eval, rr_entropy_seq_eval, beats_eval, beats_refs_eval, beat_rawlabels_eval, beat_rhythms_eval, beat_record_id_eval = \
        prepare_MITDB_data.load_data(training_set_path,
                                     seg_length=signal_length,
                                     files=files,
                                     invalid_files=invalid_files,
                                     target_fs=target_fs,
                                     train_proportion=train_proportion,
                                     denoise=denoise,
                                     rm_baseline=True,
                                     normalization=normalization,
                                     beat_labels=beat_labels,
                                     beat_group_labels=beat_group_labels,
                                     leads_names=test_leads_names,
                                     leads_number=test_leads_number,
                                     rrs_for_entropy_computing=rrs_for_entropy_computing,
                                     rrs_for_estimating_normal_rr=rrs_for_estimating_normal_rr,
                                     rrs_normalize_for_entropy=True,
                                     mm=1, r=0.05,
                                     segment_margin=50,
                                     relative_rr_scale_rate=relative_rr_scale_rate,
                                     referring_annotation_type=referring_annotation_type,
                                     annot_qrs_ext_name=annot_qrs_ext_name)

    mask_train = np.squeeze(qrs_positions_train, axis=-1)
    beatref_sequence_train = data_utils.compute_beatref_sequence(signal_length, beats_train,
                                                                 beats_refs_train, len(beat_group_labels))

    mask_eval = np.squeeze(qrs_positions_eval, axis=-1)
    beatref_sequence_eval = data_utils.compute_beatref_sequence(signal_length, beats_eval,
                                                                beats_refs_eval, len(beat_group_labels))

    inputs_train = [X_train]
    inputs_eval = [X_eval]

    if with_rrs:
        inputs_train += [preRR_train]
        inputs_eval += [preRR_eval]

    if with_entropy:
        inputs_train += [rr_entropy_seq_train]
        inputs_eval += [rr_entropy_seq_eval]

    inputs_train += [mask_train]
    inputs_eval += [mask_eval]

    dataset_train = {'X': inputs_train,
                     'Y': beatref_sequence_train}
    dataset_eval = {'X': inputs_eval,
                    'Y': beatref_sequence_eval,
                    'beats': beats_eval,
                    'beats_refs': beats_refs_eval,
                    'dataset': 'validation'}

    if class_sample_weights is not None:
        class_sample_weights = np.array(class_sample_weights, dtype=np.float32)
        class_sample_weights = np.expand_dims(class_sample_weights, axis=-1)

        sample_weights = np.matmul(Y_train, class_sample_weights)
        base_weight = np.amin(class_sample_weights)
        sample_weights[sample_weights < base_weight] = base_weight
    else:
        sample_weights = np.ones_like(Y_train[:, 0], dtype=np.float32)

    return dataset_train, dataset_eval, sample_weights


def load_test_data(test_set_path,
                   signal_length,
                   files=None,
                   invalid_files=None,
                   with_rrs=True,
                   with_entropy=True,
                   beat_labels=None,
                   beat_group_labels=None,
                   test_leads_names=None,
                   test_leads_number=1,
                   denoise=False,
                   normalization=True,
                   relative_rr_scale_rate=5,
                   rrs_for_estimating_normal_rr=100,
                   rrs_for_entropy_computing=100,
                   referring_annotation_type='beat',
                   annot_qrs_ext_name='atr',
                   target_fs=250):

    X_test, Y_test, qrs_positions_test, preRR_test, rr_entropy_seq_test, beats_test, beats_refs_test, beat_rawlabels_test, beat_rhythms_test, beat_record_id_test, \
    _, _, _, _, _, _, _, _, _, _ = \
        prepare_MITDB_data.load_data(test_set_path,
                                     seg_length=signal_length,
                                     files=files,
                                     invalid_files=invalid_files,
                                     train_proportion=1,
                                     denoise=denoise,
                                     rm_baseline=True,
                                     normalization=normalization,
                                     target_fs=target_fs,
                                     beat_labels=beat_labels,
                                     beat_group_labels=beat_group_labels,
                                     leads_names=test_leads_names,
                                     leads_number=test_leads_number,
                                     rrs_for_entropy_computing=rrs_for_entropy_computing,
                                     rrs_for_estimating_normal_rr=rrs_for_estimating_normal_rr,
                                     rrs_normalize_for_entropy=True,
                                     mm=1, r=0.05,
                                     segment_margin=50,
                                     relative_rr_scale_rate=relative_rr_scale_rate,
                                     referring_annotation_type=referring_annotation_type,
                                     annot_qrs_ext_name=annot_qrs_ext_name)

    mask_test = np.squeeze(qrs_positions_test, axis=-1)

    inputs_test = [X_test]

    if with_rrs:
        inputs_test += [preRR_test]

    if with_entropy:
        inputs_test += [rr_entropy_seq_test]

    inputs_test += [mask_test]

    print('label distribution: ')
    beats_refs_test_flat = np.concatenate(beats_refs_test, axis=0)
    labels, counts = np.unique(beats_refs_test_flat, return_counts=True)
    print('labels: ', labels)
    print('counts: ', counts)

    dataset_test = {'X': inputs_test,
                    'beats': beats_test,
                    'beats_refs': beats_refs_test,
                    'dataset': test_set_path}

    return dataset_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--cinc_path', default=None, type=str)
    parser.add_argument('--mitdb_path', default=None, type=str)
    parser.add_argument('--testset_base_path', default=None, type=str)
    parser.add_argument('--signal_length_seconds', default=20, type=float)
    parser.add_argument('--denoise', action="store_true")
    parser.add_argument('--normalization', action="store_true")
    parser.add_argument('--relative_rr_scale_rate', default=10.0, type=float)
    parser.add_argument('--rrs_for_estimating_normal_rr', default=60, type=int)
    parser.add_argument('--rrs_for_entropy_computing', default=60, type=int)
    parser.add_argument('--target_fs', default=125, type=int)

    # model structure hyper-parameters
    parser.add_argument('--supervisedMode', default='WSL', type=str)
    parser.add_argument('--ecg_filters', default=32, type=int)
    parser.add_argument('--ecg_poolinglayers', default=4, type=int)
    parser.add_argument('--ecg_kernelsize', default=8, type=int)
    parser.add_argument('--dense_units', default=16, type=int)
    parser.add_argument('--aggreg_type', default='MGMP', type=str)
    parser.add_argument('--feature_fusion_with_rrs', action="store_true")
    parser.add_argument('--feature_fusion_with_entropy', action="store_true")
    parser.add_argument('--LSE_r', default=3, type=float)
    parser.add_argument('--l2_param', default=1e-10, type=float)

    # training/testing parameters
    parser.add_argument('--run_type', default='train', type=str)    # options: train, test
    parser.add_argument('--model_id', default=None, type=str)
    parser.add_argument('--used_db_ids', default=[0, 1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--stop_mode', default='valset_early_stop', type=str)
    parser.add_argument('--min_epochs', default=20, type=int)
    parser.add_argument('--validation_split', default=0.1, type=float)
    parser.add_argument('--validate_on_whole', action="store_true")
    parser.add_argument('--training_number', default=1, type=int)
    parser.add_argument('--class_sample_weights', default=[0.1, 2, 2], nargs='+', type=float)
    parser.add_argument('--log_file', default='results/results.csv', type=str)
    parser.add_argument('--initialize_with_SL', action="store_true")
    parser.add_argument('--initialize_data_proportion', default=0.5, type=float)
    parser.add_argument('--verbose', default=0, type=int)

    # testing parameters
    parser.add_argument('--show_record_wise_results', action="store_true")

    args = parser.parse_args()

    if args.cinc_path == None:
        cinc_path = '/data1/CINC2020'
    else:
        cinc_path = args.cinc_path

    if args.mitdb_path == None:
        mitdb_path = '/data1/mitdb'
    else:
        mitdb_path = args.mitdb_path

    target_fs = args.target_fs
    signal_length = int(args.signal_length_seconds * target_fs)
    unit_length = int(2 ** 5)
    signal_length = signal_length - signal_length % unit_length + unit_length

    log_file = args.log_file
    log_folder = os.path.dirname(log_file)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder, 0o755)

    referring_annotation_type = 'beat'

    beat_labels = [['N', 'L', 'R', 'j', 'e'],
                   ['A', 'a', 'S', 'J'],
                   ['V', 'E']]

    beat_group_labels = ['N', 'S', 'V']
    snomed_ct_classes = ['426783006', '284470004', '427172004']
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'],
                          ['427172004', '17338001', '164884008']]
    persistent_label_index = np.array([1, 0, 0], dtype=np.float32)

    excluded_classes = []

    DS1_files = ['101', '106', '108', '109', '112', '114', '115', '116',
                 '118', '119', '122', '124', '201', '203', '205', '207',
                 '208', '209', '215', '220', '223', '230']
    # DS1_files = ['101', '106', '108', '109', '112', '114', '115', '116',
    #              '118', '119', '122', '124', '203', '205', '207',
    #              '208', '215', '220', '223', '230']

    DS2_files = ['100', '103', '105', '111', '113', '117', '121', '123',
                 '200', '202', '210', '212', '213', '214', '219', '221',
                 '222', '228', '231', '232', '233', '234']

    if args.testset_base_path is None:
        testset_paths = [mitdb_path]
        SL_trainingset_path = mitdb_path
    else:
        testset_base_paths = args.testset_base_path
        testset_paths = [
            os.path.join(testset_base_paths, 'mitdb'),
            os.path.join(testset_base_paths, 'mit-bih-supraventricular-arrhythmia-database-1.0.0'),
            os.path.join(testset_base_paths, 'incart/files'),
        ]
        SL_trainingset_path = mitdb_path

    training_leads = [1]

    testset_files = [
        DS2_files,
        None,
        None
    ]

    leads_names = ['II', 'MLII', 'ML2', 'I', 'MLI',  'III', 'MLIII', 'AVR', 'AVL', 'AVF',
                   'V1', 'MV1', 'V2', 'MV2', 'V3', 'MV3', 'V4', 'MV4', 'V5', 'MV5', 'V6', 'MV6',
                   'ECG', 'ECG1', 'ECG2']
    leads_number = 1

    with_rrs = args.feature_fusion_with_rrs
    with_entropy = args.feature_fusion_with_entropy

    local_output_layer_name = "predictions_beats"

    if args.run_type == 'train':
        # Run type is train
        if args.initialize_with_SL:
            initialize_data_proportion = args.initialize_data_proportion
        else:
            initialize_data_proportion = 0

        used_db_ids = args.used_db_ids
        print('used_db_ids: ', used_db_ids)

        # for args.rrs_for_estimating_normal_rr in [20, 60, 100, 120, 160]:
        #     for args.rrs_for_entropy_computing in [10, 20, 60, 100, 120]:

        if args.supervisedMode == 'WSL':
            print('Weakly supervised learning ...')

            dataset_train, dataset_eval, dataset_init, sample_weights = \
                load_training_data_for_wsl(training_path=cinc_path,
                                           signal_length=signal_length,
                                           validation_path=SL_trainingset_path,
                                           validation_files=DS1_files,
                                           with_rrs=with_rrs,
                                           with_entropy=with_entropy,
                                           beat_labels=beat_labels,
                                           beat_group_labels=beat_group_labels,
                                           snomed_ct_classes=snomed_ct_classes,
                                           equivalent_classes=equivalent_classes,
                                           training_leads=training_leads,
                                           validation_leads_names=leads_names,
                                           validation_leads_number=leads_number,
                                           denoise=args.denoise,
                                           normalization=args.normalization,
                                           class_sample_weights=args.class_sample_weights,
                                           relative_rr_scale_rate=args.relative_rr_scale_rate,
                                           rrs_for_estimating_normal_rr=args.rrs_for_estimating_normal_rr,
                                           rrs_for_entropy_computing=args.rrs_for_entropy_computing,
                                           persistent_label_index=persistent_label_index,
                                           excluded_classes=excluded_classes,
                                           referring_annotation_type=referring_annotation_type,
                                           init_data_proportion=initialize_data_proportion,
                                           annot_qrs_ext_name='atr',
                                           used_db_ids=used_db_ids,
                                           target_fs=target_fs,
                                           validate_on_whole=args.validate_on_whole)

            loss_func = 'binary_crossentropy'
        else:
            print('Fully supervised learning ...')
            dataset_train, dataset_eval, sample_weights = \
                load_training_data_for_sl(training_set_path=SL_trainingset_path,
                                          signal_length=signal_length,
                                          files=DS1_files,
                                          with_rrs=with_rrs,
                                          with_entropy=with_entropy,
                                          beat_labels=beat_labels,
                                          beat_group_labels=beat_group_labels,
                                          test_leads_names=leads_names,
                                          test_leads_number=leads_number,
                                          denoise=args.denoise,
                                          normalization=args.normalization,
                                          class_sample_weights=args.class_sample_weights,
                                          relative_rr_scale_rate=args.relative_rr_scale_rate,
                                          rrs_for_entropy_computing=args.rrs_for_entropy_computing,
                                          rrs_for_estimating_normal_rr=args.rrs_for_estimating_normal_rr,
                                          referring_annotation_type=referring_annotation_type,
                                          annot_qrs_ext_name='atr',
                                          target_fs=target_fs)

            loss_func = 'categorical_crossentropy'
            dataset_init = None

        # train
        # for args.ecg_poolinglayers in [5]:
        #     for args.ecg_kernelsize in [4, 6, 8, 10, 12]:

        ecg_poolinglayers = args.ecg_poolinglayers
        model_params = {'categories': len(beat_group_labels),
                        'channels': leads_number,
                        'supervisedMode': args.supervisedMode,
                        'ecg_filters': args.ecg_filters,
                        'ecg_poolinglayers': args.ecg_poolinglayers,
                        'ecg_kernelsize': args.ecg_kernelsize,
                        'dense_units': args.dense_units,
                        'aggreg_type': args.aggreg_type,
                        'feature_fusion_with_rrs': args.feature_fusion_with_rrs,
                        'feature_fusion_with_entropy': args.feature_fusion_with_entropy,
                        'local_output_layer_name': local_output_layer_name,
                        'LSE_r': args.LSE_r,
                        'l2_param': args.l2_param}

        print('training: ', model_params)

        for t in range(args.training_number):
            model_characteristics = {**model_params, 'try_id': t,
                                     'rrs_for_entropy_computing': args.rrs_for_entropy_computing,
                                     'rrs_for_estimating_normal_rr': args.rrs_for_estimating_normal_rr,
                                     'initialize_with_SL': args.initialize_with_SL,
                                     'initialize_data_proportion': initialize_data_proportion,
                                     'validation_split': args.validation_split,
                                     'stop_mode': args.stop_mode,
                                     'class_sample_weights': np.array_str(np.array(args.class_sample_weights)),
                                     'relative_rr_scale_rate': args.relative_rr_scale_rate}

            model_characteristics['used_db_ids'] = np.array_str(np.array(used_db_ids))
            print(model_characteristics)
            print('dataset_eval keys: ', dataset_eval.keys())
            model, timestampStr = train_wsl_model(dataset_train, dataset_eval, dataset_init,
                                                  model_params,
                                                  loss_func,
                                                  initialize_with_SL=args.initialize_with_SL,
                                                  stop_mode=args.stop_mode,
                                                  learning_rate=0.001,
                                                  model_folder='models',
                                                  epochs=100,
                                                  min_epochs=args.min_epochs,
                                                  batch_size=64,
                                                  sample_weights=sample_weights,
                                                  local_output_layer_name=local_output_layer_name,
                                                  validation_split=args.validation_split,
                                                  verbose=args.verbose)

            # validation set score
            evaluate_model(dataset_eval,
                           model,
                           model_characteristics,
                           model_folder='models',
                           beat_group_labels=beat_group_labels,
                           with_rrs=with_rrs,
                           with_qrs=False,
                           with_entropy=with_entropy,
                           timestampStr=timestampStr,
                           output_layer_name="predictions_beats")


            print('testset_paths: ', testset_paths)
            print('testset_files: ', testset_files)
            results = test_wsl_model_from_file(testset_paths,
                                               model,
                                               model_characteristics,
                                               model_folder='models',
                                               log_file=log_file,
                                               files_list=testset_files,
                                               target_fs=target_fs,
                                               beat_labels=beat_labels,
                                               beat_group_labels=beat_group_labels,
                                               with_rrs=with_rrs,
                                               with_qrs=False,
                                               with_entropy=with_entropy,
                                               timestampStr=timestampStr,
                                               output_layer_name=local_output_layer_name,
                                               seg_length=signal_length,
                                               leads_names=leads_names,
                                               leads_number=leads_number,
                                               denoise=args.denoise,
                                               normalization=args.normalization,
                                               relative_rr_scale_rate=args.relative_rr_scale_rate,
                                               rrs_for_estimating_normal_rr=args.rrs_for_estimating_normal_rr,
                                               rrs_for_entropy_computing=args.rrs_for_entropy_computing,
                                               referring_annotation_type=referring_annotation_type,
                                               show_record_wise_results=args.show_record_wise_results,
                                               save_PRC=True)

            # save results
            params = {'training_args': args}
            joblib.dump(params, os.path.join('models', 'args_'+timestampStr+'.sav'), protocol=0)

    elif args.run_type == 'test':
        # validation set score
        timestampStr = args.model_id
        # load params
        train_args_file = os.path.join('models', 'args_' + timestampStr + '.sav')
        training_args = joblib.load(train_args_file)
        args_train = training_args['training_args']

        with_rrs = args_train.feature_fusion_with_rrs
        with_entropy = args_train.feature_fusion_with_entropy

        # args_train = args

        model_characteristics = {'categories': len(beat_group_labels),
                        'channels': leads_number,
                        'supervisedMode': args_train.supervisedMode,
                        'ecg_poolinglayers': args_train.ecg_poolinglayers,
                        'ecg_kernelsize': args_train.ecg_kernelsize,
                        'aggreg_type': args_train.aggreg_type,
                        'feature_fusion_with_rrs': args_train.feature_fusion_with_rrs,
                        'feature_fusion_with_entropy': args_train.feature_fusion_with_entropy,
                        'LSE_r': args_train.LSE_r,
                        'rrs_for_entropy_computing': args_train.rrs_for_entropy_computing,
                        'rrs_for_estimating_normal_rr': args_train.rrs_for_estimating_normal_rr,
                        'initialize_with_SL': args_train.initialize_with_SL,
                        'validation_split': args_train.validation_split,
                        'stop_mode': args_train.stop_mode,
                        'used_db_ids': np.array_str(np.array(args_train.used_db_ids))}

        results = test_wsl_model_from_file(testset_paths,
                                           None,
                                           model_characteristics,
                                           model_folder='models',
                                           log_file=log_file,
                                           files_list=testset_files,
                                           target_fs=target_fs,
                                           beat_labels=beat_labels,
                                           beat_group_labels=beat_group_labels,
                                           with_rrs=with_rrs,
                                           with_qrs=False,
                                           with_entropy=with_entropy,
                                           timestampStr=timestampStr,
                                           output_layer_name=local_output_layer_name,
                                           seg_length=signal_length,
                                           leads_names=leads_names,
                                           leads_number=leads_number,
                                           denoise=args_train.denoise,
                                           normalization=args_train.normalization,
                                           relative_rr_scale_rate=args_train.relative_rr_scale_rate,
                                           rrs_for_estimating_normal_rr=args_train.rrs_for_estimating_normal_rr,
                                           rrs_for_entropy_computing=args_train.rrs_for_entropy_computing,
                                           referring_annotation_type=referring_annotation_type,
                                           show_record_wise_results=args_train.show_record_wise_results,
                                           save_PRC=False)

    exit(0)
