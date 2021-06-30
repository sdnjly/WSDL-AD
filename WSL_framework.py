"""Copyright (c) 2021 Yang Liu
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from LSEaggregation import LSEaggregation
from MaskedGlobalMaxPooling1D import MaskedGlobalMaxPooling1D


def block1D_type1(x, nb_filter, filter_len, normalization_axis, dropout=0.5, kernel_regularizer=None,
                  kernel_initializer='glorot_uniform'):
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(x)
    out = BatchNormalization(axis=normalization_axis)(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    # out = Add()([out,x],mode='sum')
    return out


def block1D_type2(x, nb_filter, filter_len, normalization_axis, dropout=0.5, kernel_regularizer=None,
                  kernel_initializer='glorot_uniform'):
    out = BatchNormalization(axis=normalization_axis)(x)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    out = BatchNormalization(axis=normalization_axis)(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(nb_filter, filter_len, padding='same',
                 kernel_regularizer=kernel_regularizer,
                 kernel_initializer=kernel_initializer)(out)
    # out = Add()([out,x],mode='sum')
    return out


def ngnet_part(inp, initfilters, initkernelsize, last_filters,
               filters_increase=False, kernel_decrease=False,
               dropout=0.5, poolinglayers=6, normalization_axis=-1,
               kernel_initializer='he_normal', kernel_regularizer=None,
               filters_increase_interpools=2, kernel_decrease_interpools=3,
               pooling_mode='max', pooling_before_merge=False):
    filters = int(initfilters)
    kernel_size = int(initkernelsize)

    hidden = Conv1D(filters, kernel_size, padding='same',
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer)(inp)
    hidden = BatchNormalization(axis=normalization_axis)(hidden)
    hidden_shortcut = Activation('relu')(hidden)

    hidden = block1D_type1(hidden_shortcut, filters, kernel_size,
                           normalization_axis, dropout,
                           kernel_regularizer,
                           kernel_initializer)
    if pooling_before_merge:
        if pooling_mode == 'ave':
            hidden = AveragePooling1D(pool_size=2, padding='valid')(hidden)
            hidden_shortcut = AveragePooling1D(pool_size=2, padding='valid')(hidden_shortcut)
        elif pooling_mode == 'conv':
            hidden = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(hidden)
            hidden_shortcut = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(hidden_shortcut)
        else:
            hidden = MaxPooling1D(pool_size=2, padding='valid')(hidden)
            hidden_shortcut = MaxPooling1D(pool_size=2, padding='valid')(hidden_shortcut)
        merge_hidden = Add()([hidden, hidden_shortcut])
    else:
        merge_hidden = Add()([hidden, hidden_shortcut])
        if pooling_mode == 'ave':
            merge_hidden = AveragePooling1D(pool_size=2, padding='valid')(merge_hidden)
        elif pooling_mode == 'conv':
            merge_hidden = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(merge_hidden)
        else:
            merge_hidden = MaxPooling1D(pool_size=2, padding='valid')(merge_hidden)

    for i in range(1, poolinglayers):

        if filters_increase and i % filters_increase_interpools == 0:
            filters += int(initfilters)
            merge_hidden = Conv1D(filters, 1, padding='same',
                                  # kernel_initializer='he_normal',
                                  kernel_regularizer=kernel_regularizer)(merge_hidden)

        if kernel_decrease and i % kernel_decrease_interpools == 0:
            kernel_size = int(kernel_size / 2)
            if kernel_size < 2:
                kernel_size = int(2)

        hidden = block1D_type2(merge_hidden, filters,
                               kernel_size, normalization_axis, dropout,
                               kernel_regularizer, kernel_initializer)
        if pooling_before_merge:
            if pooling_mode == 'ave':
                hidden = AveragePooling1D(pool_size=2, padding='valid')(hidden)
                merge_hidden = AveragePooling1D(pool_size=2, padding='valid')(merge_hidden)
            elif pooling_mode == 'conv':
                hidden = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(hidden)
                merge_hidden = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(merge_hidden)
            else:
                hidden = MaxPooling1D(pool_size=2, padding='valid')(hidden)
                merge_hidden = MaxPooling1D(pool_size=2, padding='valid')(merge_hidden)
            merge_hidden = Add()([hidden, merge_hidden])
        else:
            merge_hidden = Add()([hidden, merge_hidden])
            if pooling_mode == 'ave':
                merge_hidden = AveragePooling1D(pool_size=2, padding='valid')(merge_hidden)
            elif pooling_mode == 'conv':
                merge_hidden = Conv1D(filters, kernel_size=2, strides=2, padding='valid')(merge_hidden)
            else:
                merge_hidden = MaxPooling1D(pool_size=2, padding='valid')(merge_hidden)

    ecg_features = BatchNormalization(axis=-1)(merge_hidden)
    ecg_features = Activation('relu')(ecg_features)
    ecg_features_up = UpSampling1D(size=2 ** poolinglayers)(ecg_features)
    ecg_features_up = Conv1D(last_filters, 1, padding='same',
                             kernel_regularizer=kernel_regularizer,
                             activation='relu', )(ecg_features_up)

    return ecg_features_up


def WSL_model(categories=3, ecg_length=None,
              ecg_filters=16, ecg_kernelsize=16,
              ecg_filters_increase=False,
              ecg_kernel_decrease=False,
              ecg_dropout=0.25, ecg_poolinglayers=4,
              channels=1, dense_units=16,
              filters_increase_interpools=3,
              kernel_decrease_interpools=3,
              l2_param=1e-13,
              input_entropy_sequence=True,
              aggreg_type='MGMP',
              LSE_r=3,
              feature_fusion_with_rrs=True,
              feature_fusion_with_entropy=True,
              local_output_layer_name='predictions_beats',
              supervisedMode='WSL'):
    l2_reg = regularizers.l2(l2_param)

    with_rrs = feature_fusion_with_rrs
    with_entropy = feature_fusion_with_entropy

    ecg_input = Input(shape=(ecg_length, channels), name='ecg')
    mask_input = Input(shape=(ecg_length,), name='mask')
    rrs_input = Input(shape=(ecg_length, 1), name='rrs')

    if input_entropy_sequence:
        entropy_input = Input(shape=(ecg_length, 1), name='sample-entropy')
        sample_entropy = entropy_input
    else:
        entropy_input = Input(shape=(1,), name='sample-entropy')
        sample_entropy = RepeatVector(ecg_length)(entropy_input)

    ecg_features_map = ngnet_part(inp=ecg_input,
                                  initfilters=ecg_filters, initkernelsize=ecg_kernelsize, last_filters=dense_units,
                                  filters_increase=ecg_filters_increase, kernel_decrease=ecg_kernel_decrease,
                                  dropout=ecg_dropout, poolinglayers=ecg_poolinglayers,
                                  kernel_regularizer=l2_reg,
                                  filters_increase_interpools=filters_increase_interpools,
                                  kernel_decrease_interpools=kernel_decrease_interpools)

    if feature_fusion_with_rrs and feature_fusion_with_entropy:
        merged_features = Concatenate(axis=-1)([ecg_features_map, rrs_input, sample_entropy])
    elif feature_fusion_with_rrs:
        merged_features = Concatenate(axis=-1)([ecg_features_map, rrs_input])
    elif feature_fusion_with_entropy:
        merged_features = Concatenate(axis=-1)([ecg_features_map, sample_entropy])
    else:
        merged_features = ecg_features_map

    prediction_map = TimeDistributed(Dense(categories, kernel_regularizer=l2_reg,
                                           activation='softmax', name=local_output_layer_name))(merged_features)

    if supervisedMode == 'WSL':
        if aggreg_type == 'MGMP':
            global_prediction = MaskedGlobalMaxPooling1D()([prediction_map, mask_input], mask=mask_input)
        elif aggreg_type == 'GAP':
            global_prediction = GlobalAveragePooling1D()(prediction_map)
        elif aggreg_type == 'GMP':
            global_prediction = GlobalMaxPooling1D()(prediction_map)
        elif aggreg_type == 'LSE':
            global_prediction = LSEaggregation(r=LSE_r)([prediction_map, mask_input])
        else:
            print("supervisedMode is not valid.")
            return -1

        predictions = [global_prediction]
    else:
        predictions = [prediction_map]

    inputs = [ecg_input]
    if with_rrs:
        inputs += [rrs_input]
    if with_entropy:
        inputs += [entropy_input]

    inputs += [mask_input]

    model = Model(inputs=inputs, outputs=predictions)

    return model
