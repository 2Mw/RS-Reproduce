import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input, Conv1D
from cf.layers import crossnet, attention, mlp
from cf.utils import tensor
from cf.models.base import *


class CAN(Model):
    def __init__(self, feature_columns, cfg, directory: str = '', *args, **kwargs):
        """
        Cross & Attention Network (Temporary XD)

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory of the model.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.feature_column = feature_columns
        model_cfg = cfg['model']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.numeric_same = model_cfg['numeric_same_dim']
        self.ebd = get_embedding(self, feature_columns, self.embedding_dim, self.numeric_same, model_cfg['embedding_device'])
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'])
        self.att_layer = model_cfg['att_layer_num']
        self.attention = [
            attention.MultiheadAttention(model_cfg['att_size'], model_cfg['att_head_num'], model_cfg['att_dropout']) for
            i in range(self.att_layer)]
        self.att_trim = Dense(64, use_bias=None)
        self.final = Dense(1, 'sigmoid')

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        x = form_x(inputs, self.ebd, self.numeric_same)
        # cross part
<<<<<<< HEAD
        x = tensor.to2DTensor(x)
=======
        x = tensor.to2DTensor(embedding)
>>>>>>> exp-a
        cross_out = self.cross(x)
        # attention part
        att_x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
        for att in self.attention:
            att_x = att(att_x)
        shapes = 1
        for i in att_x.shape[1:]:
            if i is not None:
                shapes *= i
        att_out = tf.reshape(att_x, [-1, shapes])
        att_out = self.att_trim(att_out)
        out = tf.concat([cross_out, att_out], axis=-1)
        return self.final(out)
