"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the qrs_detector
(https://github.com/c-labpl/qrs_detector) project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2017 Michał Sznajder, Marta Łukowska
All rights reserved.
"""

from time import gmtime, strftime
from time import time

import math
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from scipy.signal import butter, filtfilt

import smooth

LOG_DIR = "logs/"
PLOT_DIR = "plots/"


class QRSDetectorDNN(object):
    """
    ECG QRS Detector based on the U-NET.

    The module is an implementation of QRS complex detection in the ECG signal based
    on the U-net model:
    Runnan He, Yang Liu, et al., "Automatic detection of QRS complexes using dual
    channels based on U-Net and bidirectional long short-term memory," IEEE Journal
    of Biomedical and Health Informatics, 2020.
    """

    def __init__(self, ecg_data, frequency, peak_zoom_rate=1, sigma_rate=0.1, rr_group_distance=0.2,
                 lambda_=0.5, gamma_=0.5, error_thres=1, peak_prominence=0, peak_prominence_wlen_time=0.2,
                 polarization_rate=1, max_RR_groups=10, min_new_RR_weight=0.01,
                 thres_lowing_rate_for_missed_peak=0.05, thres_lowing_rate_for_filtered_peak=1,
                 threshold_value=0.1, batch_size=1, max_seg_length=2**15, min_seg_length=2**7,
                 adaptive_std=False, punish_leak=True, use_dnn=True, models=None, pool_layers=7,
                 reverse_channel=True, normalize_signal=True, qrs_detection_method='simple_adaptive',
                 verbose=True, log_data=False, plot_data=False, show_plot=False,
                 show_reference=False, reference=None):
        """
        QRSDetectorOffline class initialisation method.
        :param string ecg_data_path: path to the ECG dataset
        :param bool verbose: flag for printing the results
        :param bool log_data: flag for logging the results
        :param bool plot_data: flag for plotting the results to a file
        :param bool show_plot: flag for showing generated results plot - will not show anything if plot is not generated
        """
        # Configuration parameters.
        # self.ecg_data_path = ecg_data_path

        self.signal_frequency = frequency  # Set ECG device frequency in samples per second here.
        frequency_scale = frequency / 250.0

        self.filter_lowcut = 5
        self.filter_highcut = 15.0
        self.filter_order = 3

        self.integration_window = int(0.15 * frequency)  # Change proportionally when adjusting frequency (in samples).

        # self.findpeaks_limit = 0.01
        self.findpeaks_spacing = int(
            50 * frequency_scale)  # Change proportionally when adjusting frequency (in samples).

        self.refractory_period = int(
            50 * frequency_scale)  # Change proportionally when adjusting frequency (in samples).
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # Loaded ECG data.
        self.ecg_data_raw = ecg_data

        # Measured and calculated values.
        self.baseline_wander_removed = None
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.normalized_signal = None
        self.model_predictions = None
        self.detected_peaks_locs = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = threshold_value

        # Detection results.
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # Final ECG data and QRS detection results array - samples with detected QRS are marked with 1 value.
        self.ecg_data_detected = None

        self.peak_zoom_rate = peak_zoom_rate
        self.sigma_rate = sigma_rate
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.error_thres = error_thres
        self.peak_prominence = peak_prominence
        self.peak_prominence_wlen = round(peak_prominence_wlen_time * self.signal_frequency)
        self.polarization_rate = polarization_rate
        self.rr_group_distance = rr_group_distance
        self.punish_leak = punish_leak
        self.adaptive_std = adaptive_std
        self.max_RR_groups = max_RR_groups
        self.min_new_RR_weight = min_new_RR_weight
        self.use_dnn = use_dnn
        self.reverse_channel = reverse_channel
        self.normalize_signal = normalize_signal
        self.pool_layers = pool_layers
        self.max_seg_length = max_seg_length
        self.min_seg_length = min_seg_length
        self.batch_size = batch_size

        if models is not None:
            self.models = models
        elif use_dnn:
            self.models = []
            # Load MITDB model
            model_structure_file = 'QRS_detector/model.json'
            model_weights_file = 'QRS_detector/weights.model'
            json_file = open(model_structure_file, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_weights_file)
            self.models.append(model)
            print('Model loaded.')
        else:
            self.models = None

        # Run whole detector flow.
        # self.load_ecg_data()
        # start = time()
        self.ecg_data_outliers_removed = self.remove_diff_outliers(self.ecg_data_raw, window=7, factor=5)
        if use_dnn:
            self.detect_peaks_dnn()
        else:
            self.detect_peaks_pt()

        if qrs_detection_method == 'adaptive':
            self.detect_qrs_adaptive_thres(thres_lowing_rate_for_missed_peak, thres_lowing_rate_for_filtered_peak)
        elif qrs_detection_method == 'simple_adaptive':
            self.detect_qrs()
        elif qrs_detection_method == 'fixed_threshold':
            self.detect_qrs_fixed_thres()
        else:
            raise ValueError("Param value for qrs_detction_method is invalid.") from Exception

        # print('time for detecting peaks: {} s'.format(str(after_detecting_peaks - start)))
        # print('time for dynamic programming: {} s'.format(str(stop - after_detecting_peaks)))

        self.show_reference = show_reference
        self.reference = reference

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            if self.use_dnn:
                self.plot_detection_data(show_plot=show_plot)
            else:
                self.plot_detection_data_pt(show_plot=show_plot)

    """Loading ECG measurements data methods."""

    def load_ecg_data(self):
        """
        Method loading ECG data set from a file.
        """
        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """ECG measurements data processing methods."""

    def remove_diff_outliers(self, sig, window, factor):
        # print('sig shape:', sig.shape)
        sig_diff = np.diff(sig)
        sig_diff_mv_median = np.convolve(sig_diff, np.ones(window) / window, mode='same')
        outlier_index = np.abs(sig_diff - sig_diff_mv_median) > factor * np.abs(sig_diff_mv_median)
        # outlier_index = np.logical_and(sig_diff > 0.2, sig_diff < -0.2)
        sig_diff[outlier_index] = sig_diff_mv_median[outlier_index]
        sig_diff = np.concatenate([[sig[0]], sig_diff])
        sig = np.cumsum(sig_diff)
        return sig

    def pred_record(self, x, max_seg_length, min_seg_length, batch_size=1):
        x = np.expand_dims(x, axis=0)
        length = x.shape[1]

        if length < max_seg_length:
            predictions = []
            valid_length = x.shape[1] - x.shape[1] % (min_seg_length)
            x = x[0:1, 0:valid_length]

            for model in self.models:
                predictions.append(model.predict(x, batch_size=batch_size).squeeze())
            x_pred = np.amax(predictions, axis=0)
        else:
            predictions = []
            seg_num = math.floor(length / max_seg_length)
            keep_length = int(seg_num * max_seg_length)
            x_segs = np.reshape(x[0, 0:keep_length], (seg_num, max_seg_length) + x.shape[2:])
            for model in self.models:
                predictions.append(model.predict(x_segs, batch_size=batch_size).squeeze())
            x_pred = np.amax(predictions, axis=0)
            x_pred = x_pred.flatten()

            predictions = []
            last_seg_length = length % max_seg_length
            if last_seg_length > min_seg_length:
                last_seg_length_valid = last_seg_length - last_seg_length%min_seg_length
                last_seg = np.expand_dims(x[0, -last_seg_length:-last_seg_length+last_seg_length_valid], axis=0)
                for model in self.models:
                    predictions.append(model.predict(last_seg, batch_size=batch_size).squeeze())
                predictions = np.amax(predictions, axis=0)
                last_seg_pred = predictions.flatten()
                x_pred = np.concatenate([x_pred, last_seg_pred], axis=0)

        return x_pred

    def detect_peaks_dnn(self):
        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        """
        # Extract measurements from loaded ECG data.
        ecg_measurements = self.ecg_data_outliers_removed

        smoothed_signal = smooth.smooth(ecg_measurements, window_len=int(self.signal_frequency), window='flat')
        ecg_measurements = ecg_measurements - smoothed_signal
        self.baseline_wander_removed = ecg_measurements

        # denoise ECG
        # DWT
        coeffs = pywt.wavedec(ecg_measurements, 'db4', level=3)
        # compute threshold
        noiseSigma = 0.01
        threshold = noiseSigma * math.sqrt(2 * math.log2(ecg_measurements.size))
        # apply threshold
        newcoeffs = coeffs
        for j in range(len(newcoeffs)):
            newcoeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
        # IDWT
        ecg_measurements = pywt.waverec(newcoeffs, 'db4')[0:len(ecg_measurements)]
        self.filtered_ecg_measurements = ecg_measurements
        self.filtered_ecg_measurements = self.filtered_ecg_measurements / (
                    np.amax(np.abs(self.filtered_ecg_measurements)) + 1e-10)

        self.differentiated_ecg_measurements = self.derivative_filter(self.filtered_ecg_measurements,
                                                                      self.signal_frequency)

        # normalize the data
        if self.normalize_signal:
            ecg_measurements = (ecg_measurements - np.mean(ecg_measurements)) / (
                        np.std(np.abs(ecg_measurements)) + 1e-10)
        self.normalized_signal = ecg_measurements

        ecg_measurements = np.expand_dims(ecg_measurements, axis=-1)
        # add reverse channel
        if self.reverse_channel:
            ecg_reverse = -1 * ecg_measurements
            ecg_measurements = np.concatenate([ecg_measurements, ecg_reverse], axis=-1)

        # get prediction of the model
        self.model_predictions = self.pred_record(ecg_measurements, max_seg_length=self.max_seg_length,
                                                  min_seg_length=self.min_seg_length, batch_size=self.batch_size)

        # # Fiducial mark - peak detection on integrated measurements.
        # candidate_locs, _ = find_peaks(self.model_predictions,
        #                                distance=self.findpeaks_spacing)
        self.detected_peaks_locs = self.findpeaks(data=self.model_predictions,
                                                  spacing=round(self.findpeaks_spacing),
                                                  candidate_locs=None,
                                                  limit=0)

        self.detected_peaks_values = self.model_predictions[self.detected_peaks_locs]

        # revise the positions
        for loc_i in range(len(self.detected_peaks_locs)):
            loc = self.detected_peaks_locs[loc_i]
            if loc > 0.075 * self.signal_frequency:
                new_loc = loc - round(0.075 * self.signal_frequency) + np.argmax(
                    ecg_measurements[loc - round(0.075 * self.signal_frequency):loc, 0])
                self.detected_peaks_locs[loc_i] = new_loc

    def detect_peaks_pt(self):
        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        """
        # Extract measurements from loaded ECG data.
        ecg_measurements = self.ecg_data_outliers_removed

        # Measurements filtering - 0-15 Hz band pass filter.
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut,
                                                              signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        # self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]
        self.filtered_ecg_measurements = self.filtered_ecg_measurements / np.amax(self.filtered_ecg_measurements)

        # Derivative - provides QRS slope information.
        # self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)
        self.differentiated_ecg_measurements = self.derivative_filter(self.filtered_ecg_measurements,
                                                                      self.signal_frequency)
        self.differentiated_ecg_measurements = self.differentiated_ecg_measurements / np.amax(
            self.differentiated_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        # self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2
        self.squared_ecg_measurements = np.abs(self.differentiated_ecg_measurements) ** self.polarization_rate

        # Moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements,
                                                       np.ones(self.integration_window),
                                                       mode='same')
        self.integrated_ecg_measurements = self.integrated_ecg_measurements / np.amax(self.integrated_ecg_measurements)

        # # Fiducial mark - peak detection on integrated measurements.
        # candidate_locs, _ = find_peaks(self.integrated_ecg_measurements,
        #                                          distance=self.findpeaks_spacing)
        self.detected_peaks_locs = self.findpeaks(data=self.integrated_ecg_measurements,
                                                  spacing=round(self.findpeaks_spacing),
                                                  candidate_locs=None,
                                                  limit=0)

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_locs]

        # find the k highest peaks
        k = round(len(self.integrated_ecg_measurements) / self.signal_frequency / 2)
        k = k if len(self.detected_peaks_values) > k else len(self.detected_peaks_values)
        largest_k_peaks = np.partition(self.detected_peaks_values, -k)[-k:]

        # remove too low peaks
        # valid_indices = self.detected_peaks_values > np.median(largest_k_peaks)/10
        # self.detected_peaks_locs = self.detected_peaks_locs[valid_indices]
        # self.detected_peaks_values = self.detected_peaks_values[valid_indices]

        # normalize the peak values
        self.detected_peaks_values = self.detected_peaks_values / np.median(largest_k_peaks)
        self.detected_peaks_values[self.detected_peaks_values > 1] = 1
        self.detected_peaks_values = self.detected_peaks_values * self.peak_zoom_rate

    """QRS detection methods."""

    def detect_qrs(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_locs, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Peak must be classified either as a noise peak or a QRS peak.
                # To be classified as a QRS peak it must exceed dynamically set threshold value.
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw)])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag)

    def detect_qrs_fixed_thres(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_locs, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Peak must be classified either as a noise peak or a QRS peak.
                # To be classified as a QRS peak it must exceed dynamically set threshold value.
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw)])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag)

    def detect_qrs_adaptive_thres(self, thres_lowing_rate_for_missed_peak=1.0, thres_lowing_rate_for_filtered_peak=1.0):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat)
        using adaptive thresholds.
        """
        # init thresholds for the integrated_ecg_measurements
        THR_SIG = 0.5
        THR_NOISE = 0.1
        SIG_LEV = THR_SIG
        NOISE_LEV = THR_NOISE
        # init thresholds for the bandpass filtered ecg
        THR_SIG1 = np.amax(self.filtered_ecg_measurements[0:2 * self.signal_frequency]) * 0.25
        THR_NOISE1 = np.mean(self.filtered_ecg_measurements[0:2 * self.signal_frequency]) * 0.5
        SIG_LEV1 = THR_SIG1
        NOISE_LEV1 = THR_NOISE1

        qrs_i = []
        qrs_c = []
        qrs_i_raw = []
        qrs_amp_raw = []
        nois_c = []
        nois_i = []
        m_selected_RR = 0
        mean_RR = 0

        for peak_id, (detected_peak_index, detected_peaks_value) in enumerate(
                zip(self.detected_peaks_locs, self.detected_peaks_values)):
            ser_back = 0
            # locate the corresponding peak in the filtered signal
            if detected_peak_index - round(0.075 * self.signal_frequency) >= 0 and \
                    detected_peak_index + round(0.075 * self.signal_frequency) <= len(
                self.filtered_ecg_measurements):
                y_i = np.amax(self.filtered_ecg_measurements[
                              detected_peak_index - round(0.075 * self.signal_frequency): \
                              detected_peak_index + round(0.075 * self.signal_frequency)])
                x_i = np.argmax(self.filtered_ecg_measurements[
                                detected_peak_index - round(0.075 * self.signal_frequency): \
                                detected_peak_index + round(0.075 * self.signal_frequency)])
            elif detected_peak_index - round(0.075 * self.signal_frequency) < 0:
                y_i = np.amax(
                    self.filtered_ecg_measurements[0: detected_peak_index + round(0.075 * self.signal_frequency)])
                x_i = np.argmax(
                    self.filtered_ecg_measurements[0: detected_peak_index + round(0.075 * self.signal_frequency)])
                ser_back = 1
            else:
                y_i = np.amax(
                    self.filtered_ecg_measurements[detected_peak_index - round(0.075 * self.signal_frequency):])
                x_i = np.argmax(
                    self.filtered_ecg_measurements[detected_peak_index - round(0.075 * self.signal_frequency):])

            # update the heart_rate (Two heart rate means one the moste recent and the other selected)
            if len(qrs_c) >= 9:
                diffRR = np.diff(qrs_i)  # calculate RR interval
                comp = qrs_i[-1] - qrs_i[-2]  # latest RR

                if m_selected_RR > 0:
                    RR_low_limit = m_selected_RR * 0.92
                    RR_high_limit = m_selected_RR * 1.16
                    stable_RR = diffRR[np.logical_and(diffRR > RR_low_limit, diffRR < RR_high_limit)]
                    if len(stable_RR) >= 8:
                        m_selected_RR = np.mean(stable_RR[-8:])
                else:
                    m_selected_RR = np.median(diffRR)

                if comp <= 0.92 * m_selected_RR or comp >= 1.16 * m_selected_RR:
                    # lower down thresholds to detect better in the integrated signal
                    THR_SIG = 0.5 * (THR_SIG)
                    # lower down thresholds to detect better in the bandpass filtered signal
                    THR_SIG1 = 0.5 * (THR_SIG1)

            # calculate the mean of the last 8 R waves to make sure that QRS is not
            # missing(If no R detected , trigger a search back) 1.66*mean
            if m_selected_RR > 0:
                test_m = m_selected_RR
            else:
                test_m = 0

            if test_m > 0:
                if (detected_peak_index - qrs_i[-1]) >= round(1.66 * test_m):  # it shows a QRS is missed

                    mediate_peaks = np.logical_and(
                        self.detected_peaks_locs > qrs_i[-1] + round(0.200 * self.signal_frequency),
                        self.detected_peaks_locs < detected_peak_index - round(0.200 * self.signal_frequency))
                    mediate_peaks_locs = self.detected_peaks_locs[mediate_peaks]
                    mediate_peaks_values = self.detected_peaks_values[mediate_peaks]
                    if len(mediate_peaks_values) > 0:
                        highest_id = np.argmax(mediate_peaks_values)
                        locs_temp = mediate_peaks_locs[highest_id]
                        pks_temp = mediate_peaks_values[highest_id]

                        if pks_temp > THR_NOISE * thres_lowing_rate_for_missed_peak:
                            qrs_c.append(pks_temp)
                            qrs_i.append(locs_temp)
                            # find the location in filtered sig
                            x_i_t = np.argmax(
                                self.filtered_ecg_measurements[locs_temp - round(0.075 * self.signal_frequency):
                                                               locs_temp + round(0.075 * self.signal_frequency)])
                            y_i_t = self.filtered_ecg_measurements[
                                locs_temp - round(0.075 * self.signal_frequency) + x_i_t]
                            # take care of bandpass signal threshold
                            if y_i_t > THR_NOISE1 * thres_lowing_rate_for_missed_peak:
                                qrs_i_raw.append(locs_temp - round(0.075 * self.signal_frequency) + x_i_t)
                                qrs_amp_raw.append(y_i_t)
                                SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1

                            not_nois = 1
                            SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV
                else:
                    not_nois = 0

            # find noise and QRS peaks
            if detected_peaks_value >= THR_SIG:
                # if a QRS candidate occurs within 360ms of the previous QRS
                # ,the algorithm determines if its T wave or QRS
                skip = 0
                if len(qrs_c) >= 3:
                    if (detected_peak_index - qrs_i[-1]) <= round(0.3600 * self.signal_frequency):
                        if detected_peak_index + round(0.075 * self.signal_frequency) > len(
                                self.differentiated_ecg_measurements):
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             qrs_i[-1] - round(0.075 * self.signal_frequency):
                                             qrs_i[-1] + round(0.075 * self.signal_frequency)])
                        elif qrs_i[-1] - round(0.075 * self.signal_frequency) < 0:
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):
                                             detected_peak_index + round(0.075 * self.signal_frequency)])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             0:qrs_i[-1] + round(0.075 * self.signal_frequency)])
                        else:
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):
                                             detected_peak_index + round(0.075 * self.signal_frequency)])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             qrs_i[-1] - round(0.075 * self.signal_frequency):
                                             qrs_i[-1] + round(0.075 * self.signal_frequency)])

                        if abs(Slope1) <= abs(0.5 * (Slope2)):  # slope less then 0.5 of previous R
                            nois_c.append(detected_peaks_value)
                            nois_i.append(detected_peak_index)
                            skip = 1  # T wave identification
                            # adjust noise level in both filtered and integrated signal
                            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                            NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV
                        else:
                            skip = 0

                if skip == 0:  # skip is 1 when a T wave is detected
                    qrs_c.append(detected_peaks_value)
                    qrs_i.append(detected_peak_index)

                #  bandpass filter check threshold
                if y_i >= THR_SIG1 * thres_lowing_rate_for_filtered_peak:
                    if ser_back:
                        qrs_i_raw.append(x_i)
                    else:
                        qrs_i_raw.append(detected_peak_index - round(0.075 * self.signal_frequency) + (x_i - 1))
                    qrs_amp_raw.append(y_i)
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1

                # adjust Signal level
                SIG_LEV = 0.125 * detected_peaks_value + 0.875 * SIG_LEV

            elif (THR_NOISE <= detected_peaks_value) and (detected_peaks_value < THR_SIG):
                # adjust Noise level in filtered sig
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                # adjust Noise level in integrated sig
                NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV

            elif detected_peaks_value < THR_NOISE:
                nois_c.append(detected_peaks_value)
                nois_i.append(detected_peak_index)

                # noise level in filtered signal
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                # noise level in integrated signal
                NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV

            #  adjust the threshold with SNR
            if NOISE_LEV != 0 or SIG_LEV != 0:
                THR_SIG = NOISE_LEV + 0.25 * (abs(SIG_LEV - NOISE_LEV))
                THR_NOISE = 0.5 * (THR_SIG)

            # adjust the threshold with SNR for bandpassed signal
            if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
                THR_SIG1 = NOISE_LEV1 + 0.25 * (abs(SIG_LEV1 - NOISE_LEV1))
                THR_NOISE1 = 0.5 * (THR_SIG1)

            skip = 0
            not_nois = 0
            ser_back = 0

        self.qrs_peaks_indices = np.array(qrs_i_raw, dtype=np.int)
        self.noise_peaks_indices = np.array(nois_i, dtype=np.int)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw)])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag)

    """Results reporting methods."""

    def print_detection_data(self):
        """
        Method responsible for printing the results.
        """
        print("qrs peaks indices")
        print(self.qrs_peaks_indices)
        print("noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """
        Method responsible for logging measured ECG and detection results to a file.
        """
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """
        Method responsible for plotting detection results.
        :param bool show_plot: flag for plotting the results and showing plot
        """

        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices, color="black"):
            axis.scatter(x=indices, y=values[indices], c=color, s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw, title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.baseline_wander_removed, title='Baseline wander removed')
        plot_data(axis=axarr[2], data=self.filtered_ecg_measurements, title='Wavelet denoised')
        plot_data(axis=axarr[3], data=self.normalized_signal, title='normalized_signal')
        plot_data(axis=axarr[4], data=self.model_predictions, title='Model predictions with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.model_predictions, indices=self.detected_peaks_locs)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:],
                  title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:], indices=self.qrs_peaks_indices)
        if self.show_reference and self.reference is not None:
            plot_points(axis=axarr[5], values=self.ecg_data_detected, indices=self.reference, color="blue")

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    def plot_detection_data_pt(self, show_plot=False):
        """
        Method responsible for plotting detection results.
        :param bool show_plot: flag for plotting the results and showing plot
        """

        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices, color="black"):
            axis.scatter(x=indices, y=values[indices], c=color, s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw, title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements,
                  title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.detected_peaks_locs)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:],
                  title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:], indices=self.qrs_peaks_indices)
        if self.show_reference and self.reference is not None:
            plot_points(axis=axarr[5], values=self.ecg_data_detected, indices=self.reference, color="blue")

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    """Tools methods."""

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
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
        b, a = butter(filter_order, [low, high], btype="bandpass", output='ba')
        y = filtfilt(b, a, data)
        return y

    def derivative_filter(self, data, signal_freq):
        # print(data.shape)
        if signal_freq != 200:
            int_c = (5 - 1) / (signal_freq / 40)
            b = np.interp(np.arange(1, 5.1, int_c), np.arange(1, 5.1),
                          np.array([1, 2, 0, -2, -1]) * (1 / 8) * signal_freq)
            # print(b)
        else:
            b = np.array([1, 2, 0, -2, -1]) * signal_freq / 8

        filted_data = filtfilt(b, 1, data)
        return filted_data

    def findpeaks(self, data, spacing=1, candidate_locs=None, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        if candidate_locs is not None:
            peak_candidate[candidate_locs] = True
        else:
            peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c >= h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        else:
            limit = np.mean(data[ind]) / 2
            ind = ind[data[ind] > limit]

        return ind


if __name__ == "__main__":
    # fs_ = 500
    # ecg_path = '../dataset/CPSC2019/data/data_00691'
    # ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
    # ref_path = '../dataset/CPSC2019/ref/R_00691'
    # reference = sio.loadmat(ref_path)['R_peak'].flatten()

    # models = []
    # for i in range(5):
    #     model_structure_file = 'model_varyLRTrue_unet_uselstmFalse_16filters_9pools_kernel7_drop0.2/model.json'
    #     model_weights_file = 'model_varyLRTrue_unet_uselstmFalse_16filters_9pools_kernel7_drop0.2/model_' + str(
    #         i) + '.model'
    #     json_file = open(model_structure_file, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     model = model_from_json(loaded_model_json)
    #     model.load_weights(model_weights_file)
    #     models.append(model)

    beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x', '/']
    fs_ = 400
    ecg_path = '../../TrainingSet/data/A04'
    # ecg_data, fields = wfdb.srdsamp(ecg_path, channels=[0])
    # ecg_data = ecg_data.squeeze()
    # ann = wfdb.rdann(ecg_path, 'atr')
    # r_ref = [round(ann.annsamp[i]) for i in range(len(ann.annsamp)) if ann.anntype[i] in beat_labels_all]
    # r_ref = np.array(r_ref)
    # r_ref = r_ref[(r_ref >= 0.5 * fs_) & (r_ref <= len(ecg_data) - 0.5 * fs_)]

    ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
    ecg_data = ecg_data[round(2.62e5):round(2.65e5)]
    print('ecg_data shape: ', ecg_data.shape)

    params = {
        'peak_zoom_rate': 1,
        'sigma_rate': 0.2,
        'lambda_': 0.7,
        'gamma_': 0.5,
        'peak_prominence': 0.0,
        'polarization_rate': 1,
        'rr_group_distance': 0.2,
        'models': None,
        'punish_leak': True,
        'adaptive_std': False,
        'max_RR_groups': 10,
        'use_dnn': True,
        'normalize_signal': False,
        'reverse_channel': True,
        'pool_layers': 9,
        'qrs_detection_method': 'fixed_threshold',
        'thres_lowing_rate_for_missed_peak': 0.05,
        'thres_lowing_rate_for_filtered_peak': 0.05,
        'plot_data': True,
        'threshold_value': 0.1
    }

    start = time()
    with tf.device('/cpu:0'):
        qrs_detector = QRSDetectorDNN(ecg_data=ecg_data, frequency=fs_, **params)
    end = time()
    print('Running time: %s Seconds' % (end - start))

    # result_mat = {
    #     'raw_signal': ecg_data,
    #     'integrated_ecg_measurements': qrs_detector.integrated_ecg_measurements,
    #     'detected_peaks_locs':qrs_detector.detected_peaks_locs,
    #     'detected_peaks_values':qrs_detector.detected_peaks_values,
    #     'qrs_peaks_indices':qrs_detector.qrs_peaks_indices,
    #     'ref_qrs_indices': r_ref
    # }

    result_mat = {
        'raw_signal': ecg_data,
        'model_predictions': qrs_detector.model_predictions,
        'detected_peaks_locs': qrs_detector.detected_peaks_locs,
        'detected_peaks_values': qrs_detector.detected_peaks_values,
        'qrs_peaks_indices': qrs_detector.qrs_peaks_indices,
        # 'ref_qrs_indices': r_ref
    }

    logpath = "{:s}QRS_offline_detector_result_{:s}.mat".format('logs/',
                                                                strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    sio.savemat(logpath, result_mat)
