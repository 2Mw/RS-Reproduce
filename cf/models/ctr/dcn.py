import tensorflow as tf
from keras.layers import Dense
from cf.models.ctr.base import get_embedding, model_summary, form_x
from cf.layers import crossnet, mlp, linear, gate
from cf.utils import tensor
from cf.models.ctr.cowclip import Cowclip
from cf.models.ctr.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class DCN(Cowclip):
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
            super(DCN, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DCN, self).__init__(*args, **kwargs)
        # Optional params
        self.use_embed_gate = model_cfg['use_embed_gate']
        # model layers
        self.linear = linear.Linear(feature_columns)
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.cross_net = crossnet.CrossNet(self.layer_num, model_cfg['cross_w_reg'], model_cfg['cross_b_reg'])
        self.mlp = mlp.MLP(self.hidden_units, self.activation, self.dnn_dropout)
        self.dense_final = Dense(1, activation=None)
        if self.use_embed_gate:
            self.embed_gate = gate.EmbeddingGate(self.sparse_len, self.embedding_dim)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # x = tf.concat([self.ebd[f](v) if f[0] == 'C' else tf.expand_dims(v, 1) for f, v in inputs.items()], axis=1)
        sparse_x, dense_x = form_x(inputs, self.ebd, True)
        # Embedding Gate
        if self.use_embed_gate:
            sparse_x = self.embed_gate(sparse_x)
        x = tf.concat([sparse_x, dense_x], axis=-1)
        x = tensor.to2DTensor(x)
        # Cross Network
        cross_x = self.cross_net(x)
        # DNN
        dnn_x = self.mlp(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        y = self.dense_final(total_x)
        if self.linear_res:
            linear_out = self.linear(inputs)
            y = y + linear_out
        return tf.nn.sigmoid(y)
