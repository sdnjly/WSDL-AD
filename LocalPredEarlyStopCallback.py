import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model

import validation_scores


class LocalPredEarlyStopCallback(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

       Arguments:
          patience: Number of epochs to wait after min has been hit. After this
          number of no improvement, training stops.
    """

    def __init__(self, patience=0, min_epochs=20, local_output_layer_name=None, validation_data=None,
                 validation_beatref=None, validation_beatindexes=None, best_model_file=None,
                 verbose=0):
        super(LocalPredEarlyStopCallback, self).__init__()
        self.patience = patience
        self.min_epochs = min_epochs
        self.local_output_layer_name = local_output_layer_name
        self.validation_data = validation_data
        self.validation_beatref = validation_beatref
        self.validation_beatindexes = validation_beatindexes
        self.best_model_file = best_model_file
        self.verbose = verbose
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = -1 * np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # current = logs.get("loss")
        # get local model
        local_prediction_layer = self.model.get_layer(self.local_output_layer_name)
        model_local = Model(inputs=self.model.inputs,
                            outputs=[local_prediction_layer.output])
        # predict validation set
        validation_pred = model_local.predict(self.validation_data)
        beat_predictions = validation_scores.convert_beat_prediction2(validation_pred,
                                                                      self.validation_beatindexes,
                                                                      beat_preQRS_length=1,
                                                                      beat_postQRS_length=1)
        beat_predictions = np.concatenate(beat_predictions, axis=0)
        beat_predictions = beat_predictions.argmax(axis=1).astype(np.float).squeeze()
        beat_annotations = np.concatenate(self.validation_beatref, axis=0)
        conf_mat = confusion_matrix(beat_annotations, beat_predictions)
        TP, FP, FN, Se, PPv, Acc, Sp, F1 = validation_scores.get_se_ppv_acc_from_confmat(conf_mat)
        current = np.mean(F1)
        if self.verbose:
            print("Validation scores: ", F1)

        if np.greater(current, self.best):
            if self.verbose:
                print("validation score updated from {} to {}".format(self.best, current))
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > self.min_epochs:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

        # save best weight to file
        self.model.save_weights(self.best_model_file)

