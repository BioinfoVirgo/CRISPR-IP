import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import to_categorical

def transformIO(xtrain, xtest, ytrain, ytest, seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1, seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0], 1, seq_len, coding_dim)
    input_shape = (1, seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape

def crispr_ip(xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+crispr_ip.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1,input_shape[-1]), padding='valid', data_format='channels_first', kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        model.save('{}+crispr_ip.h5'.format(saved_prefix))
    model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model