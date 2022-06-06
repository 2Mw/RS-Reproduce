import time
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from cf.models.base import get_embedding, form_x


class Linear(Layer):
    def __init__(self, feature_column, use_bias=False, initializer=None, **kwargs):
        super().__init__(**kwargs)
        # params
        self.seed = time.time_ns()
        self.feature_column = feature_column
        self.reg = keras.regularizers.l2(1e-5)
        self.initializer = keras.initializers.glorot_normal(self.seed) if initializer is None else initializer
        # networks
        self.ebd = get_embedding(feature_column, 1, prefix='linear')
        self.lr = Dense(1, use_bias=use_bias, kernel_initializer=self.initializer)

    def call(self, inputs, *args, **kwargs):
        sparse_x, dense_x = form_x(inputs, self.ebd, True)
        fc = self.lr(dense_x)
        out = tf.reshape(tf.reduce_sum(sparse_x, axis=-1), [-1, 1]) + fc
        return out
