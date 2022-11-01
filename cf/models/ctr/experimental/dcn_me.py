import tensorflow as tf
from keras.layers import Dense
from cf.models.base import get_embedding, model_summary, form_x
from cf.layers import crossnet, mlp, gate, moe
from cf.utils import tensor
from cf.models.ctr.cowclip import Cowclip
from cf.models.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat
from tensorflow import keras


class DCNME(Cowclip):
    def __init__(self, feature_columns, config, directory: str = '', *args, **kwargs):
        """
        Deep & Cross Network

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory of the model.
        :param args:
        :param kwargs:
        """
        # model params
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.feature_column = feature_columns
        self.hidden_units = model_cfg['hidden_units']
        self.layer_num = len(self.hidden_units)
        self.dnn_dropout = model_cfg['dropout']
        self.activation = model_cfg['activation']
        self.embedding_dim = model_cfg['embedding_dim']
        self.linear_res = model_cfg['linear_res']
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)))
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(DCNME, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DCNME, self).__init__(*args, **kwargs)
        # Optional params
        x_dim = self.sparse_len * self.embedding_dim + len(feature_columns) - self.sparse_len
        initializer = keras.initializers.he_normal
        # broker params
        self.bridge_type = model_cfg['bridge_type']
        self.broker_experts = model_cfg['broker_experts']
        self.broker_gates = model_cfg['broker_gates']
        # model layers
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.cross_net = [crossnet.CrossNet(1, model_cfg['cross_w_reg'], model_cfg['cross_b_reg'], initializer) for _ in
                          range(self.layer_num)]
        self.mlp = [mlp.MLP([x_dim], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                            model_cfg['use_residual'], initializer=initializer) for _ in range(self.layer_num)]
        self.dense_final = Dense(1, activation=None)
        # bridge and broker
        self.bridges = [gate.BridgeModule(self.bridge_type) for _ in range(self.layer_num)]
        self.brokers = [moe.MMOE(self.broker_experts, [x_dim], self.broker_gates, dropout=model_cfg['dropout'],
                                 use_bn=True, residual=False) for _ in range(self.layer_num)]

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # x = tf.concat([self.ebd[f](v) if f[0] == 'C' else tf.expand_dims(v, 1) for f, v in inputs.items()], axis=1)
        x = form_x(inputs, self.ebd, False)
        x = tensor.to2DTensor(x)
        cross_x, dnn_x = self.brokers[0](x)
        cross_0 = cross_x
        bridge_x = None
        for i in range(self.layer_num):
            cross_x = self.cross_net[i](cross_x, cross_0)
            dnn_x = self.mlp[i](dnn_x)
            bridge_x = self.bridges[i]([cross_x, dnn_x])
            if i + 1 < self.layer_num:
                cross_x, dnn_x = self.brokers[i + 1](bridge_x)

        out = tf.concat([cross_x, dnn_x], axis=-1)
        return tf.nn.sigmoid(self.dense_final(out))
