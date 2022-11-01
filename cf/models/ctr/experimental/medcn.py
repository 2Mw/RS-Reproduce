import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers import crossnet, mlp, gate, moe
from cf.utils import tensor
from cf.models.base import get_embedding, form_x, model_summary
from cf.preprocess.feature_column import SparseFeat, SequenceFeat
from cf.models.ctr.cowclip import Cowclip
from cf.models.base import checkCowclip
from tensorflow import keras


class MEDCN(Cowclip):
    def __init__(self, feature_columns, cfg, directory: str = '', *args, **kwargs):
        """
        Multi-mixture of experts Deep & Cross net.

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory to save the model.
        :param args:
        :param kwargs:
        """
        # model params
        self.feature_column = feature_columns
        model_cfg = cfg['model']
        train_cfg = cfg['train']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.sparse_len = len(
            list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, SequenceFeat), feature_columns)))
        initializer = keras.initializers.he_normal
        # dcn-m params
        low_rank = model_cfg['low_rank']
        num_experts = model_cfg['num_experts']
        cross_layers = self.cross_layers = model_cfg['cross_layers']
        l2_reg_cross = model_cfg['l2_reg_cross']
        # cowclip params
        checkCowclip(self, train_cfg['cowclip'])
        if train_cfg['cowclip']:
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(MEDCN, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(MEDCN, self).__init__(*args, **kwargs)
        # broker params
        self.bridge_type = model_cfg['bridge_type']
        self.broker_experts = model_cfg['broker_experts']
        self.broker_gates = model_cfg['broker_gates']
        # Optional params
        x_dim = self.sparse_len * self.embedding_dim + len(feature_columns) - self.sparse_len
        self.seq_split = model_cfg.get('seq_split')
        # model layers
        # embedding part
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        # parallel part
        self.cross = [crossnet.CrossNetMix(low_rank, num_experts, 1, l2_reg_cross, initializer=initializer) for _ in
                      range(cross_layers)]
        self.mlp = [mlp.MLP([x_dim], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                            model_cfg['use_residual'], initializer=initializer) for _ in range(cross_layers)]
        # bridge and broker
        self.bridges = [gate.BridgeModule(self.bridge_type) for _ in range(cross_layers)]
        self.brokers = [moe.MMOE(self.broker_experts, [x_dim], self.broker_gates, dropout=model_cfg['dropout'],
                                 use_bn=True, residual=False) for _ in range(cross_layers)]
        self.using_embedding_broker = model_cfg['using_embedding_broker']
        self.using_feature_broker = model_cfg['using_feature_broker']

        # final dense
        self.final_dense = Dense(1, activation=None)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        x = form_x(inputs, self.ebd, False, seq_split=self.seq_split)
        x = tensor.to2DTensor(x)
        if self.using_embedding_broker:
            cross_x, dnn_x = self.brokers[0](x)
        else:
            cross_x = dnn_x = x
        cross_0 = cross_x
        bridge_x = None
        for i in range(self.cross_layers):
            cross_x = self.cross[i](cross_x, cross_0)
            dnn_x = self.mlp[i](dnn_x)
            if self.using_feature_broker:
                bridge_x = self.bridges[i]([cross_x, dnn_x])
                if i + 1 < self.cross_layers:
                    cross_x, dnn_x = self.brokers[i + 1](bridge_x)

        out = tf.concat([cross_x, dnn_x], axis=-1)
        return tf.nn.sigmoid(self.final_dense(out))
