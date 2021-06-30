"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import math
import os

import numpy as np
import pywt
import wfdb
from pyentrp import entropy as ent
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

import smooth
from data_utils import extract_relative_RR_feature, extract_RR_entropy, preprocess


def compute_RR_sampleEn_single(qrs_times, fs=250, normalize=False,
                               mm=1, r=0.1):
    qrs_times = qrs_times / fs
    if len(qrs_times) > 2:
        rrs = np.diff(qrs_times)
    else:
        rrs = np.array([1])

    if normalize:
        rrs = rrs / np.median(rrs)

    if len(rrs) > 3:
        sample_entropy = ent.sample_entropy(rrs, mm, r)
        if sample_entropy > 5:
            sample_entropy = 5
    else:
        sample_entropy = 0

    return sample_entropy


def load_data(input_path, seg_length, train_proportion=0.5,
              denoise=False,
              rm_baseline=False, files=None,
              invalid_files=None,
              normalization=False,
              target_fs=250,
              beat_labels=None,
              beat_group_labels=None,
              leads_names=None,
              leads_number=1,
              rrs_for_entropy_computing=20,
              rrs_for_estimating_normal_rr=20,
              rrs_normalize_for_entropy=True,
              mm=1, r=0.05,
              segment_margin=50,
              segment_strade_in_beat=0,
              relative_rr_scale_rate=5,
              referring_annotation_type='beat',
              annot_ext_name='atr',
              annot_qrs_ext_name='atr',
              filtering_by_rhythms=None):

    if invalid_files is None:
        invalid_files = []

    beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'S', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x']

    if len(beat_labels) is None:
        beat_labels = [['N', 'L', 'R', 'j', 'e'],
                       ['A', 'a', 'S', 'J'],
                       ['V', 'E'],
                       ['F'],
                       ['f', 'Q']]

        beat_group_labels = ['N', 'S', 'V', 'F', 'Q']

    if leads_names is None:
        leads_names = ['I', 'MLI', 'II', 'MLII', 'ML2', 'III', 'MLIII', 'AVR', 'AVL', 'AVF',
                 'V1', 'MV1', 'V2', 'MV2', 'V3', 'MV3', 'V4', 'MV4', 'V5', 'MV5', 'V6', 'MV6',
                 'ECG', 'ECG1', 'ECG2']


    # get record list in the input path
    if files is None:
        files = glob.glob(os.path.join(input_path, '*.hea'))
        files.sort()
    else:
        for i in range(len(files)):
            files[i] = os.path.join(input_path, files[i])

    ecg_train = []
    qrs_map_train = []
    relative_rr_map_train = []
    rr_entropy_map_train = []
    refs_train = []
    beatlist_train = []
    beatref_train = []
    beat_rawlabels_train = []
    beat_rhythms_train = []
    record_id_train = []

    ecg_test = []
    qrs_map_test = []
    relative_rr_map_test = []
    rr_entropy_map_test = []
    refs_test = []
    beatlist_test = []
    beatref_test = []
    beat_rawlabels_test = []
    beat_rhythms_test = []
    record_id_test = []

    # load data and extract features of each record one by one
    for file in files:
        filename = os.path.splitext(file)[0]
        basename = os.path.basename(filename)
        record_id = basename
        # print(filename)
        if basename in invalid_files:
            continue

        ecg_segs, qrs_map_segs, relative_rr_map_segs, rr_entropy_map_segs, \
        refs_segs, beatlist_segs, beatref_segs, beat_rawlabels_segs, \
        beat_rhythms_segs, record_id_segs = \
        load_file(filename, seg_length, train_proportion,
                  denoise, rm_baseline, normalization,
                  target_fs, beat_labels, beat_group_labels,
                  beat_labels_all, leads_names, leads_number, rrs_for_entropy_computing,
                  rrs_for_estimating_normal_rr,
                  rrs_normalize_for_entropy,
                  mm, r, segment_margin, relative_rr_scale_rate,
                  referring_annotation_type, annot_ext_name,
                  annot_qrs_ext_name, filtering_by_rhythms=filtering_by_rhythms,
                  segment_strade=segment_strade_in_beat)

        seg_nums = len(ecg_segs)
        training_segs = int(seg_nums * train_proportion)

        ecg_train.extend(ecg_segs[:training_segs])
        qrs_map_train.extend(qrs_map_segs[:training_segs])
        relative_rr_map_train.extend(relative_rr_map_segs[:training_segs])
        rr_entropy_map_train.extend(rr_entropy_map_segs[:training_segs])
        refs_train.extend(refs_segs[:training_segs])
        beatlist_train.extend(beatlist_segs[:training_segs])
        beatref_train.extend(beatref_segs[:training_segs])
        beat_rawlabels_train.extend(beat_rawlabels_segs[:training_segs])
        beat_rhythms_train.extend(beat_rhythms_segs[:training_segs])
        record_id_train.extend(record_id_segs[:training_segs])

        if training_segs < seg_nums:
            ecg_test.extend(ecg_segs[training_segs:])
            qrs_map_test.extend(qrs_map_segs[training_segs:])
            relative_rr_map_test.extend(relative_rr_map_segs[training_segs:])
            rr_entropy_map_test.extend(rr_entropy_map_segs[training_segs:])
            refs_test.extend(refs_segs[training_segs:])
            beatlist_test.extend(beatlist_segs[training_segs:])
            beatref_test.extend(beatref_segs[training_segs:])
            beat_rawlabels_test.extend(beat_rawlabels_segs[training_segs:])
            beat_rhythms_test.extend(beat_rhythms_segs[training_segs:])
            record_id_test.extend(record_id_segs[training_segs:])


    ecg_train = pad_sequences(ecg_train, maxlen=seg_length, padding='post', truncating='post', dtype='float32')
    ecg_test = pad_sequences(ecg_test, maxlen=seg_length, padding='post', truncating='post', dtype='float32')

    qrs_map_train = pad_sequences(qrs_map_train, maxlen=seg_length, padding='post', truncating='post', dtype='float32')
    qrs_map_test = pad_sequences(qrs_map_test, maxlen=seg_length, padding='post', truncating='post', dtype='float32')

    relative_rr_map_train = pad_sequences(relative_rr_map_train, maxlen=seg_length, padding='post', truncating='post',
                                          dtype='float32')
    relative_rr_map_test = pad_sequences(relative_rr_map_test, maxlen=seg_length, padding='post', truncating='post',
                                         dtype='float32')

    rr_entropy_map_train = pad_sequences(rr_entropy_map_train, maxlen=seg_length, padding='post', truncating='post',
                                         dtype='float32')
    rr_entropy_map_test = pad_sequences(rr_entropy_map_test, maxlen=seg_length, padding='post', truncating='post',
                                        dtype='float32')

    refs_train = np.array(refs_train, dtype='float32')
    refs_test = np.array(refs_test, dtype='float32')

    return ecg_train, refs_train, qrs_map_train, relative_rr_map_train, rr_entropy_map_train, \
           beatlist_train, beatref_train, beat_rawlabels_train, beat_rhythms_train, record_id_train, \
           ecg_test, refs_test, qrs_map_test, relative_rr_map_test, rr_entropy_map_test, \
           beatlist_test, beatref_test, beat_rawlabels_test, beat_rhythms_test, record_id_test


def load_file(filename, seg_length, train_proportion=0.5,
              denoise=False,
              rm_baseline=False,
              normalization=False,
              target_fs=250,
              beat_labels=None,
              beat_group_labels=None,
              beat_labels_all=None,
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
              annot_qrs_ext_name='atr',
              filtering_by_rhythms=None,
              segment_strade=0):

    ecg_segs = []
    qrs_map_segs = []
    relative_rr_map_segs = []
    rr_entropy_map_segs = []
    refs_segs = []
    beatlist_segs = []
    beatref_segs = []
    beat_rawlabels_segs = []
    beat_rhythms_segs = []
    record_id_segs = []

    group_num = len(beat_group_labels)
    basename = os.path.basename(filename)
    record_id = basename

    beat_notes = []
    beat_samples = []
    beat_samples_all = []
    beat_raw_labels = []
    beat_rhythms = []

    # load annotation
    ann = wfdb.rdann(filename, annot_ext_name)
    ann_qrs = wfdb.rdann(filename, annot_qrs_ext_name)

    j = -1
    rhythm = ''
    for i in range(len(ann_qrs.symbol)):
        while j < len(ann.sample) - 1 and ann_qrs.sample[i] >= ann.sample[j + 1]:
            j += 1
            if ann.aux_note[j].startswith('('):
                rhythm = ann.aux_note[j][1:].strip('\x00')

        symbol = ann_qrs.symbol[i].strip('\x00')
        if symbol in beat_labels_all:
            beat_samples_all.append(ann_qrs.sample[i])

            if referring_annotation_type == 'rhythm':
                for group_id in range(len(beat_labels)):
                    if rhythm in beat_labels[group_id]:
                        beat_notes.append(group_id)
                        beat_samples.append(ann_qrs.sample[i])
                        beat_raw_labels.append(symbol)
                        beat_rhythms.append(rhythm)
            else:
                for group_id in range(len(beat_labels)):
                    if symbol in beat_labels[group_id]:
                        beat_notes.append(group_id)
                        beat_samples.append(ann_qrs.sample[i])
                        beat_raw_labels.append(symbol)
                        beat_rhythms.append(rhythm)

    beat_samples = np.array(beat_samples, dtype=np.int)
    beat_notes = np.array(beat_notes, dtype=np.int)
    beat_samples_all = np.array(beat_samples_all, dtype=np.int)
    beat_number = beat_samples_all.shape[0]

    # print('beat number: ', beat_number)
    # print('interesting beat number: ', len(beat_samples))

    # load ECG
    signals, fields = wfdb.rdsamp(filename)

    # select channel according to the order in leads_names
    lead_ids = []
    signal_leads = fields['sig_name']
    for i in range(len(leads_names)):
        if leads_names[i] in signal_leads:
            lead_ids.append(signal_leads.index(leads_names[i]))
            if len(lead_ids) >= leads_number:
                break

    # if no enough leads are found, add other leads of the signal
    if len(lead_ids) < leads_number:
        print('no enough leads found.')
        for i in range(len(signal_leads)):
            if i not in lead_ids:
                lead_ids.append(i)
                if len(lead_ids) >= leads_number:
                    break

    signals = signals[:, lead_ids]

    if len(signals.shape) < 2:
        signals = np.expand_dims(signals, axis=-1)

    input_fs = fields['fs']
    # print('input fs: ', input_fs)
    if 0 < target_fs != input_fs:
        step = round(input_fs / target_fs)
        assert (step > 0), "Ratio between input_fs and target_fs is too low."
        signal_length = signals.shape[0]
        signals = signals[0:signal_length:step, :]
        beat_samples = np.floor(beat_samples / step).astype(dtype=np.int)
        beat_samples_all = np.floor(beat_samples_all / step).astype(dtype=np.int)

        # output_length = int(signals.shape[0] * target_fs / input_fs)
        # signals = resample(signals, output_length)
        # beat_samples = np.floor(beat_samples * target_fs / input_fs).astype(dtype=np.int)
        # beat_samples_all = np.floor(beat_samples_all * target_fs / input_fs).astype(dtype=np.int)
    signal_length = signals.shape[0]

    # # pre-processing
    signals = preprocess(signals, rm_baseline, denoise, target_fs)

    # extract features
    # QRS positions
    beat_samples_all = beat_samples_all[beat_samples_all < signal_length]
    qrs_postion_map = np.zeros((signal_length, 1), dtype=np.float32)
    qrs_postion_map[beat_samples_all] = 1

    # relative RR feature
    relative_rrs, relative_rrs_map = extract_relative_RR_feature(beat_samples_all,
                                                                 signal_length,
                                                                 rrs_for_estimating_normal_rr,
                                                                 relative_rr_scale_rate)

    # RR entropy
    rrs_sample_entropy, rrs_sample_entropy_map = extract_RR_entropy(beat_samples_all,
                                                                    signal_length,
                                                                    rrs_for_entropy_computing,
                                                                    rrs_normalize_for_entropy,
                                                                    mm,
                                                                    r,
                                                                    target_fs)

    seg_startpoint = 0
    pre_segment_last_beat_id = -1
    # print('signal_length: ', signal_length)
    # print('beat_samples number: ', len(beat_samples))
    while seg_startpoint < signal_length and pre_segment_last_beat_id < len(beat_samples) - 1:
        seg_startpoint = beat_samples[pre_segment_last_beat_id + 1] - segment_margin
        seg_startpoint = 0 if seg_startpoint < 0 else seg_startpoint
        seg_endpoint = seg_startpoint + seg_length if seg_startpoint + seg_length < signal_length else signal_length

        # segment ECG
        ecg_segment = signals[seg_startpoint:seg_endpoint]

        if len(ecg_segment) < target_fs:
            break

        # pre-processing
        # ecg_segment = preprocess(ecg_segment, rm_baseline, denoise, target_fs)

        if normalization:
            # normalize the data
            scaler = StandardScaler()
            scaler.fit(ecg_segment)
            ecg_segment = scaler.transform(ecg_segment)

        # segment features maps
        qrs_map_segment = qrs_postion_map[seg_startpoint:seg_endpoint]
        relative_rrs_map_segment = relative_rrs_map[seg_startpoint:seg_endpoint]
        rrs_sample_entropy_map_segment = rrs_sample_entropy_map[seg_startpoint:seg_endpoint]

        # segment beat list
        seg_beat_ids = np.where(np.logical_and(beat_samples > seg_startpoint,
                                               beat_samples + segment_margin < seg_startpoint + seg_length))[0]
        labels_segment = beat_notes[seg_beat_ids]

        if labels_segment.size == 0:
            # print('no label segment.')
            continue

        if segment_strade > 0:
            pre_segment_last_beat_id += segment_strade
        else:
            pre_segment_last_beat_id = seg_beat_ids[-1]

        labels = np.unique(labels_segment)
        label_onehot = np.zeros((group_num,))
        for i in range(group_num):
            if i in labels:
                label_onehot[i] = 1

        beats_segment = beat_samples[seg_beat_ids]
        beats_segment = beats_segment - seg_startpoint

        raw_labels_segment = [beat_raw_labels[b] for b in seg_beat_ids]
        raw_rhythm_segment = [beat_rhythms[b] for b in seg_beat_ids]

        if filtering_by_rhythms is not None:
            common = False
            for rhythm in raw_rhythm_segment:
                if rhythm in filtering_by_rhythms:
                    common = True
                    break

            if not common:  # skip the segment if no common rhythms
                continue

        ecg_segs.append(ecg_segment)
        qrs_map_segs.append(qrs_map_segment)
        relative_rr_map_segs.append(relative_rrs_map_segment)
        rr_entropy_map_segs.append(rrs_sample_entropy_map_segment)
        refs_segs.append(label_onehot)
        beatlist_segs.append(beats_segment)
        beatref_segs.append(labels_segment)
        beat_rawlabels_segs.append(raw_labels_segment)
        beat_rhythms_segs.append(raw_rhythm_segment)
        record_id_segs.append(record_id)

    return ecg_segs, qrs_map_segs, relative_rr_map_segs, rr_entropy_map_segs, \
           refs_segs, beatlist_segs, beatref_segs, beat_rawlabels_segs, \
           beat_rhythms_segs, record_id_segs
