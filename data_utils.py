import numpy as np
import math
import pywt
from scipy.signal import butter, filtfilt

import entropy as ent
"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import smooth


def preprocess(signals, rm_baseline, denoise, fs, lowcut=0.1, highcut=30):
    if rm_baseline:
        for i in range(signals.shape[1]):
            smoothed_signal = smooth.smooth(signals[:, i], window_len=int(fs), window='flat')
            signals[:, i] = signals[:, i] - smoothed_signal

    if denoise:
        signals = bandpass_filter(signals, lowcut, highcut, fs, 1)

        # # denoise ECG
        # for i in range(signals.shape[1]):
        #     # DWT
        #     coeffs = pywt.wavedec(signals[:, i], 'db4', level=2)
        #     # compute threshold
        #     noiseSigma = 0.1
        #     threshold = noiseSigma * math.sqrt(2 * math.log2(signals[:, i].size))
        #     # apply threshold
        #     newcoeffs = coeffs
        #     for j in range(len(newcoeffs)):
        #         newcoeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
        #
        #     # IDWT
        #     signals[:, i] = pywt.waverec(newcoeffs, 'db4')[0:len(signals)]

    return signals


def extract_relative_RR_feature(qrs_positions,
                                signal_length,
                                rrs_for_estimating_normal_rr,
                                relative_rr_scale_rate):

    beat_number = qrs_positions.shape[0]
    relative_rrs = np.zeros((beat_number,), dtype=np.float32)
    relative_rrs_map = np.zeros((signal_length, 1), dtype=np.float32)
    for i in range(1, beat_number):
        qrs_range_begin = i - int(rrs_for_estimating_normal_rr / 2)
        qrs_range_begin = qrs_range_begin if qrs_range_begin > 0 else 0
        qrs_range_end = qrs_range_begin + rrs_for_estimating_normal_rr
        qrs_range_end = qrs_range_end if qrs_range_end < len(qrs_positions) else len(qrs_positions)
        qrs_range_begin = qrs_range_end - rrs_for_estimating_normal_rr
        qrs_range_begin = qrs_range_begin if qrs_range_begin > 0 else 0

        beats_in_context = qrs_positions[qrs_range_begin: qrs_range_end]
        context_rr = np.sort(np.diff(beats_in_context))
        context_normal_rr = context_rr[-len(context_rr)//2:].mean()
        rr = qrs_positions[i] - qrs_positions[i - 1]
        relative_rrs[i] = (context_normal_rr - rr) / context_normal_rr * relative_rr_scale_rate

        # map feature to QRS position
        range_begin = math.ceil((qrs_positions[i - 1] + qrs_positions[i]) / 2)
        range_end = signal_length if i == len(qrs_positions) - 1 else math.ceil(
            (qrs_positions[i + 1] + qrs_positions[i]) / 2)
        relative_rrs_map[range_begin:range_end, 0] = relative_rrs[i]

    return relative_rrs, relative_rrs_map


def extract_RR_entropy(qrs_positions,
                       signal_length,
                       rrs_for_entropy_computing,
                       rrs_normalize_for_entropy,
                       mm,
                       r,
                       fs):
    beat_number = qrs_positions.shape[0]
    rrs_sample_entropy = np.zeros((beat_number,), dtype=np.float32)
    for i in range(beat_number):
        qrs_range_begin = i - int(rrs_for_entropy_computing / 2)
        qrs_range_begin = qrs_range_begin if qrs_range_begin > 0 else 0
        qrs_range_end = qrs_range_begin + rrs_for_entropy_computing
        qrs_range_end = qrs_range_end if qrs_range_end < len(qrs_positions) else len(qrs_positions)
        qrs_range_begin = qrs_range_end - rrs_for_entropy_computing
        qrs_range_begin = qrs_range_begin if qrs_range_begin > 0 else 0

        beats_for_entropy = qrs_positions[qrs_range_begin: qrs_range_end]
        rrs_sample_entropy[i] = compute_RR_sampleEn_single(beats_for_entropy, fs,
                                                           normalize=rrs_normalize_for_entropy,
                                                           mm=mm, r=r)

    rrs_sample_entropy_map = compute_feature_map(qrs_positions, rrs_sample_entropy, signal_length)

    return rrs_sample_entropy, rrs_sample_entropy_map


def compute_RR_sampleEn_single(qrs_positions,
                               fs=250,
                               normalize=False,
                               mm=1, r=0.1):
    qrs_positions = qrs_positions / fs
    if len(qrs_positions) > 2:
        rrs = np.diff(qrs_positions)
    else:
        rrs = np.array([1])

    if normalize:
        rrs = rrs / np.median(rrs)

    # r = r * np.std(rrs)

    if len(rrs) > 3:
        try:
            sample_entropy = ent.sample_entropy(rrs, mm, r)
            if sample_entropy > 5:
                sample_entropy = 5
        except Exception as e:
            print(e)
            print('QRS number:', len(qrs_positions))
            sample_entropy = 5
    else:
        sample_entropy = 5

    return sample_entropy


def compute_beatref_sequence(segment_length, beats_list, beat_labels, class_num):
    segment_num = len(beats_list)
    beatref_sequence = np.zeros((segment_num, segment_length, class_num), dtype=np.float32)
    for i in range(segment_num):
        qrs_times = beats_list[i]
        for j in range(len(qrs_times)):
            class_id = beat_labels[i][j]
            beatref_sequence[i, qrs_times[j], class_id] = 1

    return beatref_sequence


def compute_feature_map(qrs_positions, features, signal_length):
    if len(features.shape) < 2:
        features = np.expand_dims(features, axis=-1)
    feature_map = np.zeros((signal_length, features.shape[1]), dtype=np.float32)
    beat_number = len(qrs_positions)
    for i in range(beat_number):
        # map feature to QRS position
        range_begin = 0 if i == 0 else math.ceil((qrs_positions[i - 1] + qrs_positions[i]) / 2)
        range_end = signal_length if i == len(qrs_positions) - 1 else math.ceil(
            (qrs_positions[i + 1] + qrs_positions[i]) / 2)
        feature_map[range_begin:range_end, :] = features[i, :]

    return feature_map


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    data_filtered = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[-1]):
        b, a = butter(filter_order, [low, high], btype="bandpass", output='ba')
        data_filtered[:, i] = filtfilt(b, a, data[:, i])

    return data_filtered


# function to generate all the sub lists
def sub_lists (l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j: i])
    return lists