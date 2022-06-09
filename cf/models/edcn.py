import tensorflow as tf
from tensorflow import keras
from cf.models.cowclip import Cowclip
from keras.models import Model
from keras.layers import Dense
from cf.models.base import get_embedding, form_x, model_summary
from cf.layers import crossnet, mlp, linear, gate
from cf.utils.tensor import to2DTensor
from cf.models.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class EDCN(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        """
        The model of Enhancing Deep & Cross.

        link: https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf

        Tips: The output dimension of every corresponding layer of cross and mlp are same.

        :param feature_column:
        :param config:
        :param directory:
        :param args:
        :param kwargs:
        """
        # model params
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.feature_column = feature_column
        self.embedding_dim = model_cfg['embedding_dim']
        self.linear_res = model_cfg['linear_res']
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_column)))
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(EDCN, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(EDCN, self).__init__(*args, **kwargs)
        # Optional params
        x_dim = self.sparse_len * self.embedding_dim + len(feature_column) - self.sparse_len
        self.num_cross_layer = model_cfg['cross_layers']
        initializer = keras.initializers.he_normal
        # model layers
        # embedding layer
        self.ebd = get_embedding(feature_column, self.embedding_dim, model_cfg['embedding_device'])
        # parallel structure
        self.mlp = [mlp.MLP([x_dim], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                            model_cfg['use_residual'], initializer) for _ in range(self.num_cross_layer)]
        self.cross = [crossnet.CrossNet(1) for _ in range(self.num_cross_layer)]
        # bridge and regulation
        self.bridges = [gate.BridgeModule(x_dim, model_cfg['bridge_type']) for _ in range(self.num_cross_layer)]
        self.regulations = [gate.RegulationModule(len(feature_column), self.embedding_dim, self.sparse_len,
                                                  model_cfg['tau']) for _ in range(self.num_cross_layer)]
        self.final_dense = Dense(1, activation=None)
        self.linear = linear.Linear(feature_column)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # bridge module and regularization module.
        x = form_x(inputs, self.ebd, False)
        x = to2DTensor(x)
        cross_x, dnn_x = self.regulations[0](x)
        cross_0 = cross_x
        bridge_x = None
        for i in range(self.num_cross_layer):
            cross_x = self.cross[i](cross_x, cross_0)
            dnn_x = self.mlp[i](dnn_x)
            bridge_x = self.bridges[i]([cross_x, dnn_x])
            if i + 1 < self.num_cross_layer:
                cross_x, dnn_x = self.regulations[i + 1](bridge_x)

        out = tf.concat([cross_x, dnn_x, bridge_x], axis=-1)
        return tf.nn.sigmoid(self.final_dense(out))
