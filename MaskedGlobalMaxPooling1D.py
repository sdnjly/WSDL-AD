"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer


class MaskedGlobalMaxPooling1D(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)

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
            x = x * mask
        return K.max(x, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0][0], input_shape[0][2]
