"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from datetime import datetime
from os import path
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam

import WSL_framework
import validation_scores
from LocalPredEarlyStopCallback import LocalPredEarlyStopCallback


def train_wsl_model(dataset_train, dataset_valid, dataset_train_sl,
                    model_params,
                    loss,
                    initialize_with_SL=False,
                    learning_rate=0.001,
                    model_folder='models',
                    epochs=50,
                    min_epochs=20,
                    batch_size=32,
                    sample_weights=None,
                    stop_mode='valset_early_stop',
                    validation_split=0.1,
                    verbose=0,
                    local_output_layer_name="predictions_beats"):
    if not path.exists(model_folder):
        os.mkdir(model_folder, 0o755)

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H%M%S%f")

    model_structure_file = path.join(model_folder, 'model_' + timestampStr + '.json')

    # construct model
    K.clear_session()

    model = WSL_framework.WSL_model(**model_params)

    if model_params['supervisedMode'] == 'WSL' and initialize_with_SL:
        print('Initialize model with supervised learning on small DB ...')
        local_prediction_layer = model.get_layer(model_params['local_output_layer_name'])
        model_local = Model(inputs=model.inputs,
                            outputs=[local_prediction_layer.output])

        adam = Adam(lr=learning_rate)
        model_local.compile(loss='categorical_crossentropy',
                            optimizer=adam)

        model_local.fit(dataset_train_sl['X'], dataset_train_sl['Y_local'],
                        batch_size=batch_size,
                        epochs=10,
                        shuffle=True,
                        verbose=verbose)

    adam = Adam(lr=learning_rate)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['binary_accuracy'])

    init_weights = model.get_weights()

    model_json = model.to_json()
    with open(model_structure_file, "w") as json_file:
        json_file.write(model_json)

    log_file = path.join(model_folder, 'log_' + timestampStr + '.csv')
    best_model_file = path.join(model_folder, 'weights_' + timestampStr + '.model')

    csv_logger = CSVLogger(log_file)
    print('dataset_eval keys: ', dataset_valid.keys())

    model.set_weights(init_weights)

    if stop_mode == 'fix_epochs':
        model.fit(dataset_train['X'], dataset_train['Y'],
                  batch_size=batch_size,
                  epochs=epochs,
                  sample_weight=sample_weights,
                  shuffle=True,
                  callbacks=[csv_logger],
                  verbose=verbose
                  )
        model.save_weights(best_model_file)
    elif stop_mode == 'split_early_stop':
        model.fit(dataset_train['X'], dataset_train['Y'],
                  batch_size=batch_size,
                  epochs=epochs,
                  sample_weight=sample_weights,
                  validation_split=validation_split,
                  shuffle=True,
                  callbacks=[csv_logger],
                  verbose=verbose
                  )
    else:
        early_stopping = LocalPredEarlyStopCallback(patience=10, min_epochs=min_epochs,
                                                    local_output_layer_name=local_output_layer_name,
                                                    validation_data=dataset_valid['X'],
                                                    validation_beatref=dataset_valid['beats_refs'],
                                                    validation_beatindexes=dataset_valid['beats'],
                                                    best_model_file=best_model_file,
                                                    verbose=verbose)
        model.fit(dataset_train['X'], dataset_train['Y'],
                  batch_size=batch_size,
                  epochs=epochs,
                  sample_weight=sample_weights,
                  shuffle=True,
                  callbacks=[csv_logger, early_stopping],
                  verbose=verbose
                  )

    return model, timestampStr
