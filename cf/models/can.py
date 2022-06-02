import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers import crossnet, attention
from cf.utils import tensor
from cf.models.base import get_embedding, form_x, model_summary
from cf.preprocess.feature_column import SparseFeat
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip


class CAN(Cowclip):
    def __init__(self, feature_columns, cfg, directory: str = '', *args, **kwargs):
        """
        Cross & Attention Network (Temporary XD)

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory of the model.
        :param args:
        :param kwargs:
        """
        # model params
        self.feature_column = feature_columns
        model_cfg = cfg['model']
        train_cfg = cfg['train']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)))
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(CAN, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(CAN, self).__init__(*args, **kwargs)
        # att params
        self.att_size = model_cfg['att_size']
        self.head_num = model_cfg['att_head_num']
        self.att_layer = model_cfg['att_layer_num']
        # self.units = model_cfg['hidden_units']
        # model layers
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'])
        self.attention = [attention.MultiheadAttention(self.att_size, self.head_num, model_cfg['att_dropout']) for
                          i in range(self.att_layer)]
        self.att_trim = Dense(64, use_bias=None)
        self.final = Dense(1, 'sigmoid')

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        sparse_x, dense_x = form_x(inputs, self.ebd, True)
        # cross part
        cross_x = tf.concat([sparse_x, dense_x], axis=-1)
        x = tensor.to2DTensor(cross_x)
        cross_out = self.cross(x)
        # attention part
        att_x = tf.reshape(sparse_x, [-1, self.sparse_len, self.embedding_dim])
        for att in self.attention:
            att_x = att(att_x)
        att_x = tensor.to2DTensor(att_x)
        att_out = self.att_trim(att_x)
        out = tf.concat([cross_out, att_out], axis=-1)
        return self.final(out)
