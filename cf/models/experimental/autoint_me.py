import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers.attention import MultiheadAttention
from cf.models.base import get_embedding, model_summary, form_x
from cf.layers import linear, mlp, gate, moe
from cf.preprocess.feature_column import SparseFeat
from cf.utils.tensor import to2DTensor
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip
from tensorflow import keras


class AutoIntME(Cowclip):
    def __init__(self, feature_columns, config, directory: str = "", *args, **kwargs):
        """
        AutoInt model

        :param config: The configuration of the hyper-parameters.
        :param feature_columns: The feature columns.
        :param directory: The output directory.
        """
        # parameters
        model_cfg = config['model']
        train_cfg = config['train']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.feature_column = feature_columns
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)))
        self.linear_res = model_cfg['linear_res']
        # if numeric_same_dim = True, means model can take numeric feature into calculation directly.
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(AutoIntME, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(AutoIntME, self).__init__(*args, **kwargs)
        # att params
        self.att_size = model_cfg['att_size']
        self.head_num = model_cfg['att_head_num']
        self.att_layer = model_cfg['att_layer_num']
        self.units = model_cfg['hidden_units']
        # Optional params
        x_dim = self.sparse_len * self.embedding_dim + len(feature_columns) - self.sparse_len
        initializer = keras.initializers.he_normal
        # broker params
        self.bridge_type = model_cfg['bridge_type']
        self.broker_experts = model_cfg['broker_experts']
        self.broker_gates = model_cfg['broker_gates']
        # networks
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.attention = [MultiheadAttention(self.att_size, self.head_num) for i in range(self.att_layer)]
        self.final = Dense(1, use_bias=False)
        self.mlp = [mlp.MLP([x_dim], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                            model_cfg['use_residual'], initializer=initializer) for _ in range(self.att_layer)]
        # bridge and broker
        self.bridges = [gate.BridgeModule(self.bridge_type) for _ in range(self.att_layer)]
        self.brokers = [moe.MMOE(self.broker_experts, [x_dim], self.broker_gates, dropout=model_cfg['dropout'],
                                 use_bn=True, residual=False) for _ in range(self.att_layer)]

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # Attention
        x = form_x(inputs, self.ebd, False, self.numeric_same_dim)

        att_x, dnn_x = self.brokers[0](x)
        bridge_x = None
        for i in range(self.att_layer):
            att_x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
            att_x = to2DTensor(self.attention[i](att_x))
            dnn_x = self.mlp[i](dnn_x)
            bridge_x = self.bridges[i]([att_x, dnn_x])
            if i + 1 < self.att_layer:
                att_x, dnn_x = self.brokers[i + 1](bridge_x)

        out = to2DTensor(tf.concat([att_x, dnn_x], axis=-1))
        return tf.nn.sigmoid(self.final(out))
