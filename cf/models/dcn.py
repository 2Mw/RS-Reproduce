import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from cf.models.base import *
from cf.layers import crossnet, mlp
from cf.utils import tensor


class DCN(Model):
    def __init__(self, feature_columns, config, directory: str = '', *args, **kwargs):
        """
        Deep & Cross Network

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory of the model.
        :param args:
        :param kwargs:
        """
        super(DCN, self).__init__(*args, **kwargs)
        self.directory = directory
        model_cfg = config['model']
        self.feature_column = feature_columns
        self.hidden_units = model_cfg['hidden_units']
        self.layer_num = len(self.hidden_units)
        self.dnn_dropout = model_cfg['dropout']
        self.activation = model_cfg['activation']
        self.embedding_dim = model_cfg['embedding_dim']
        self.numeric_same = model_cfg['numeric_same_dim']
        self.ebd = get_embedding(self, feature_columns, self.embedding_dim, self.numeric_same, model_cfg['embedding_device'])
        self.cross_net = crossnet.CrossNet(self.layer_num, model_cfg['cross_w_reg'], model_cfg['cross_b_reg'])
        self.mlp = mlp.MLP(self.hidden_units, self.activation, self.dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # x = tf.concat([self.ebd[f](v) if f[0] == 'C' else tf.expand_dims(v, 1) for f, v in inputs.items()], axis=1)
        x = form_x(inputs, self.ebd, self.numeric_same)
        x = tensor.to2DTensor(x)
        # Cross Network
        cross_x = self.cross_net(x)
        # DNN
        dnn_x = self.mlp(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        # TODO 未加正则化
        return tf.nn.sigmoid(self.dense_final(total_x))
