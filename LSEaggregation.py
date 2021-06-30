"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer


class LSEaggregation(Layer):

    def __init__(self, r=0, **kwargs):
        self.supports_masking = True
        self.r = r
        super(LSEaggregation, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        x = x[0]
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0, 2, 1])
            # to make the masked values in x be equal to zero
            weighted_x = K.sum(K.exp(self.r * x) * mask, axis=1) / (K.sum(mask, axis=1) + K.epsilon())
            return 1 / self.r * K.log(weighted_x + K.epsilon()) * K.max(mask)
        else:
            weighted_x = K.mean(K.exp(self.r * x), axis=1)
            return 1 / self.r * K.log(weighted_x + K.epsilon())

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0][0], input_shape[0][2]

    def get_config(self):
        config = {
            'r': self.r,
        }
        base_config = super(LSEaggregation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
