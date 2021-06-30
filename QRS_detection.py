import argparse
import os
import numpy as np
import scipy.io as sio
from tensorflow.keras.models import model_from_json

import prepare_CINC_data
import QRSDetectorDNN

parser = argparse.ArgumentParser()
parser.add_argument('--cinc_path', default=None, type=str)
args = parser.parse_args()

cinc_main_path = args.cinc_path

# load data
cinc_paths = [
    os.path.join(cinc_main_path, 'CPSC/Training_WFDB'),
    os.path.join(cinc_main_path, 'CPSC2/Training_2'),
    os.path.join(cinc_main_path, 'PTB-XL/WFDB'),
    os.path.join(cinc_main_path, 'E/WFDB')
]

models = []
model_structure_file = 'QRS_detector/model.json'
model_weights_file = 'QRS_detector/weights.model'
json_file = open(model_structure_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)
models.append(model)

params = {
    'use_dnn': True,
    'pool_layers': 7,
    'models': models,
    'reverse_channel': True,
    'max_seg_length': 2 ** 20,
    'min_seg_length': 2 ** 7,
    'batch_size': 1,
    'qrs_detection_method': 'fixed_threshold',
    'threshold_value': 0.1,
    'verbose': False,
    'plot_data': False,
    'show_plot': False
}

fs = 500
lead = 1

for input_path in cinc_paths:
    qrs_dir = os.path.join(input_path, 'qrs_indexes')
    if not os.path.exists(qrs_dir):
        os.mkdir(qrs_dir, 0o755)

    for f in os.listdir(input_path):
        header_file = os.path.join(input_path, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(header_file):

            recording, header_data = prepare_CINC_data.load_challenge_data(header_file)
            data = np.transpose(recording)
            signal_length = data.shape[0]
            signal = data[:, lead]

            # get sampling frequency of the file
            record_fs = int(header_data[0].split(' ')[2])

            # get resolution
            rs = int(header_data[1].split(' ')[2].split('/')[0])
            signal = signal / rs

            if record_fs != fs:
                step = round(record_fs / fs)
                signal_length = signal_length - signal_length % step
                signal = signal[0:signal_length:step]

                signal_length = round(signal_length / step)
            else:
                step = 1

            qrsdetector = QRSDetectorDNN.QRSDetectorDNN(signal,
                                                        fs,
                                                        **params)
            qrs_indexes = qrsdetector.qrs_peaks_indices.squeeze()
            qrs_indexes = qrs_indexes * step

            qrs_file = os.path.join(qrs_dir, f[:-4] + '.mat')
            sio.savemat(qrs_file, {'qrs': qrs_indexes})
