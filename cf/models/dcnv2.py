import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from cf.models.base import *
from cf.layers import crossnet, mlp


class DCNv2(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        super(DCNv2, self).__init__(*args, **kwargs)
        self.directory = directory
        model_cfg = config['model']
        self.feature_column = feature_column
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'],
                           model_cfg['use_bn'], initializer=keras.initializers.he_normal)
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'], initializer=keras.initializers.he_normal)

        self.embedding_dim = model_cfg['embedding_dim']
        self.numeric_same = model_cfg['numeric_same_dim']
        self.ebd = get_embedding(self, feature_column, self.embedding_dim, self.numeric_same, model_cfg['embedding_device'])
        # self.final_dense = Dense(1, model_cfg['activation'])
        self.final_dense = Dense(1, activation=None)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # 对于 类别型数据使用 embedding，对于数值型数值不使用 embedding
        x = form_x(inputs, self.ebd, self.numeric_same)
        cross_out = self.cross(x)
        dnn_out = self.mlp(x)
        total_x = tf.concat([cross_out, dnn_out], axis=-1)
        return tf.nn.sigmoid(self.final_dense(total_x))
