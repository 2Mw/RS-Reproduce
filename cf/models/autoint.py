import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dense
from keras.regularizers import l2
from cf.layers.attention import MultiheadAttention
from cf.models.base import *


class AutoInt(Model):
    def __init__(self, feature_columns, config, directory: str = "", *args, **kwargs):
        """
        AutoInt model

        :param config: The configuration of the hyper-parameters.
        :param feature_columns: The feature columns.
        :param directory: The output directory.
        """
        super(AutoInt, self).__init__(*args, **kwargs)
        model_cfg = config['model']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.feature_column = feature_columns
        self.numeric_same = model_cfg['numeric_same_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim, self.numeric_same, model_cfg['embedding_device'])
        self.att_layer = model_cfg['att_layer_num']
        self.attention = [MultiheadAttention(model_cfg['att_size'], model_cfg['att_head_num']) for i in
                          range(self.att_layer)]
        self.final = Dense(1, activation='sigmoid')

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        x = form_x(inputs, self.ebd, self.numeric_same)
        # 对于注意力机制层需要将shape修改为 (batch, future, embedding)
        x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
        for att in self.attention:
            x = att(x)
        shapes = 1
        for i in x.shape[1:]:
            if i is not None:
                shapes *= i
        x = tf.reshape(x, [-1, shapes])
        out = self.final(x)
        return out
