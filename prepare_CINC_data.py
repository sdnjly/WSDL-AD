"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from random import randint, random

import numpy as np
import pandas
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.models import model_from_json

from QRSDetectorDNN import QRSDetectorDNN
from data_utils import preprocess, extract_relative_RR_feature, extract_RR_entropy


def load_data_simple(input_path, header_files=None, target_length=2000,
                     fs=250, leads=None, classes=None,
                     necessary_classes=None,
                     excluded_classes=None,
                     resample_signal=True, remove_bw=True,
                     denoise=True, normalization=False,
                     normal_selecting_rate=0.1, equivalent_classes=None,
                     keep_ratio_of_N=1,
                     rrs_for_estimating_normal_rr=20,
                     relative_rr_scale_rate=5,
                     rrs_for_entropy_computing=20,
                     rrs_normalize_for_entropy=True,
                     mm=1,
                     r=0.05,
                     persistent_label_index=None):

    if header_files is None:
        header_files = []
        for f in os.listdir(input_path):
            g = os.path.join(input_path, f)
            if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
                header_files.append(g)
    else:
        for i in range(len(header_files)):
            header_files[i] = os.path.join(input_path, header_files[i])

    if classes is None:
        classes = ['426783006', '164909002', '713427006', '284470004', '427172004']
    if equivalent_classes is None:
        equivalent_classes = [['713427006', '59118001'],
                              ['284470004', '63593006'],
                              ['427172004', '17338001'],
                              ['427172004', '17338001']]

    # load DNN models for QRS detection
    models = []
    # model_structure_file = 'QRS_detector/model.json'
    # model_weights_file = 'QRS_detector/weights.model'
    # json_file = open(model_structure_file, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(model_weights_file)
    # models.append(model)

    X = []
    Y = []
    QRSs = []
    X_qrs_map = []
    X_relative_RR_feature_map = []
    X_RR_entropy = []
    file_names = []
    raw_labels = []
    lengthes = []

    for hf in header_files:
        recording, header_data = load_challenge_data(hf)
        labels = []
        for iline in header_data:
            if iline.startswith('#Dx'):
                labels = iline.split(': ')[1].split(',')
                for lb in range(len(labels)):
                    labels[lb] = labels[lb].strip()
                break
        labels = replace_equivalent_classes(labels, equivalent_classes)

        intersec_1 = set(labels) & set(necessary_classes)
        if len(necessary_classes) > 0 and len(intersec_1) == 0:
            continue

        intersec_2 = set(labels) & set(classes)
        if len(intersec_2) == 0 or (len(intersec_2) == 1 and '426783006' in intersec_2):
            # if this record doesn't belong to the considered classes or is N, then down sampling
            if random() > normal_selecting_rate:
                continue

        if excluded_classes is not None:
            intersec_3 = set(labels) & set(excluded_classes)
            if len(intersec_3) > 0:
                continue

        fs_record = int(header_data[0].split(' ')[2])
        signal_length = int(header_data[0].split(' ')[3])
        # if signal_length / fs_record * fs > target_length:
        #     continue

        signal, label_vector, qrs_indices, labels = load_file(hf,
                                                              target_fs=fs,
                                                              target_length=target_length,
                                                              classes=classes,
                                                              leads=leads,
                                                              equivalent_classes=equivalent_classes,
                                                              resample_signal=resample_signal,
                                                              remove_bw=remove_bw,
                                                              denoise=denoise,
                                                              normalize=normalization,
                                                              models=models)
        # extract features
        # QRS positions
        qrs_postion_map = np.zeros((signal_length, 1), dtype=np.float32)
        if len(qrs_indices) > 0:
            qrs_postion_map[qrs_indices] = 1

        # relative RR feature
        relative_rrs, relative_rrs_map = extract_relative_RR_feature(qrs_indices,
                                                                     signal_length,
                                                                     rrs_for_estimating_normal_rr,
                                                                     relative_rr_scale_rate)

        # RR entropy
        rrs_sample_entropy, rrs_sample_entropy_map = extract_RR_entropy(qrs_indices,
                                                                        signal_length,
                                                                        rrs_for_entropy_computing,
                                                                        rrs_normalize_for_entropy,
                                                                        mm,
                                                                        r,
                                                                        fs)

        # sub-sampling N recordings
        if np.sum(label_vector) == 1 and label_vector[0] == 1 and random() > keep_ratio_of_N:
            continue

        if persistent_label_index is not None:
            # when the labels on persistent_label_index are all zero, no SVTA nor IVR, then add N to it
            if (label_vector * persistent_label_index).sum() == 0 and (not '426761007' in labels) and (not '49260003' in labels):
                label_vector[0] = 1

        if len(qrs_indices) > 0:
            qrs_indices = qrs_indices[qrs_indices < target_length]

        # add the sample to the set
        X.append(signal)
        Y.append(label_vector)
        X_qrs_map.append(qrs_postion_map)
        X_relative_RR_feature_map.append(relative_rrs_map)
        X_RR_entropy.append(rrs_sample_entropy_map)
        QRSs.append(qrs_indices)
        file_names.append(hf)
        raw_labels.append(labels)
        lengthes.append(len(signal))

    # pad signals and features to the same length
    if len(X) > 0:
        X = sequence.pad_sequences(X, maxlen=target_length, dtype='float32', padding='post', truncating='post')
        X_qrs_map = sequence.pad_sequences(X_qrs_map, maxlen=target_length, dtype='float32', padding='post', truncating='post')
        X_relative_RR_feature_map = sequence.pad_sequences(X_relative_RR_feature_map, maxlen=target_length, dtype='float32', padding='post', truncating='post')
        X_RR_entropy = sequence.pad_sequences(X_RR_entropy, maxlen=target_length, dtype='float32', padding='post', truncating='post')
        Y = np.array(Y, dtype=np.float32)

    return X, Y, X_qrs_map, X_relative_RR_feature_map, X_RR_entropy, QRSs, file_names, raw_labels, lengthes


def load_file(header_file,
              target_fs=250,
              target_length=12800,
              classes=None,
              equivalent_classes=None,
              leads=None,
              resample_signal=True,
              remove_bw=True,
              denoise=True,
              normalize=True,
              models=None):

    recording, header_data = load_challenge_data(header_file)
    data = np.transpose(recording)
    signal_length = data.shape[0]

    if leads:
        signal = data[:, leads]
    else:
        signal = data

    label_vector = np.zeros(len(classes))
    labels = []
    for iline in header_data:
        if iline.startswith('#Dx'):
            labels = iline.split(': ')[1].split(',')
            for lb in range(len(labels)):
                labels[lb] = labels[lb].strip()
            break

    if equivalent_classes is not None:
        labels = replace_equivalent_classes(labels, equivalent_classes)

    for k, c in enumerate(classes):
        if c in labels:
            label_vector[k] = 1

    # get sampling frequency of the file
    fs = int(header_data[0].split(' ')[2])

    # get resolution
    rs = int(header_data[1].split(' ')[2].split('/')[0])
    signal = signal / rs

    # resample the signal to target frequency
    if fs == 257:
        fs = 250
    if fs != target_fs and resample_signal:
        step = round(fs / target_fs)
        signal_length = signal_length - signal_length % step
        signal = signal[0:signal_length:step, :]
    else:
        step = 1

    # preprocess
    signal = preprocess(signal, remove_bw, denoise, target_fs)

    # normalize
    if normalize:
        # normalize the data
        scaler = StandardScaler()
        scaler.fit(signal)
        signal = scaler.transform(signal)

    # detect qrs
    # qrs_detector = QRSDetectorDNN(ecg_data=signal[:, 0],
    #                               verbose=False,
    #                               frequency=target_fs,
    #                               use_dnn=True,
    #                               pool_layers=7,
    #                               models=models,
    #                               reverse_channel=True,
    #                               qrs_detection_method='fixed_threshold',
    #                               threshold_value=0.1,
    #                               log_data=False,
    #                               plot_data=False,
    #                               show_plot=False,
    #                               show_reference=False)
    # qrs_indices = qrs_detector.qrs_peaks_indices

    file_dir, base_name = os.path.split(header_file)
    qrs_file_name = os.path.join(file_dir, 'qrs_indexes', base_name[:-4] + '.mat')
    try:
        qrs_indices = sio.loadmat(qrs_file_name)['qrs'][0]
        qrs_indices = qrs_indices // step
    except Exception as e:
        print(e)
        print(qrs_file_name)
        qrs_indices = np.array([])

    return signal, label_vector, qrs_indices, labels


# Find unique classes.
def get_classes(filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0]  # Use the first class as the representative class.
    return classes


# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = sio.loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header


def random_pad(x, target_length):
    if len(x) > target_length:
        begin = randint(0, len(x) - target_length)
        x_pad = x[begin:begin + target_length]
    else:
        begin = randint(0, target_length - len(x))
        x_pad = np.zeros((target_length,) + x.shape[1:])
        x_pad[begin:begin + len(x)] = x
    return x_pad


def random_crop(x, precentage=1, target_length=0):
    if target_length <= 0:
        target_length = round(len(x) * precentage)
    begin = randint(0, len(x) - target_length)
    x_pad = x[begin:begin + target_length]

    return x_pad


def get_class_map(files):
    class_dict = {}
    for file in files:
        data_frame = pandas.read_csv(file, header=0)
        class_dict.update(dict(zip(data_frame.Code, data_frame.Abbreviation)))

    return class_dict
