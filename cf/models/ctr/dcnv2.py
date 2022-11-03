import tensorflow as tf
from tensorflow import keras
from cf.models.ctr.cowclip import Cowclip
from keras.models import Model
from keras.layers import Dense
from cf.models.ctr.base import get_embedding, form_x, model_summary
from cf.layers import crossnet, mlp, linear, gate
from cf.utils.tensor import to2DTensor
from cf.models.ctr.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class DCNv2(Cowclip):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        # model params
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.feature_column = feature_column
        self.embedding_dim = model_cfg['embedding_dim']
        self.linear_res = model_cfg['linear_res']
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_column)))
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(DCNv2, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DCNv2, self).__init__(*args, **kwargs)
        # Optional params
        self.use_embed_gate = model_cfg['use_embed_gate']
        # model layers
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'],
                           model_cfg['use_bn'], initializer=keras.initializers.he_normal)
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'], initializer=keras.initializers.he_normal)
        self.ebd = get_embedding(feature_column, self.embedding_dim, model_cfg['embedding_device'])
        self.linear = linear.Linear(feature_column)
        self.final_dense = Dense(1, activation=None)
        if self.use_embed_gate:
            self.embed_gate = gate.EmbeddingGate(self.sparse_len, self.embedding_dim)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # 对于 类别型数据使用 embedding，对于数值型数值不使用 embedding
        sparse_x, dense_x = form_x(inputs, self.ebd, True)
        # Embedding Gate
        if self.use_embed_gate:
            sparse_x = self.embed_gate(sparse_x)
        x = tf.concat([sparse_x, dense_x], axis=-1)
        x = to2DTensor(x)
        cross_out = self.cross(x)
        dnn_out = self.mlp(x)
        total_x = tf.concat([cross_out, dnn_out], axis=-1)
        y = self.final_dense(total_x)
        if self.linear_res:
            linear_out = self.linear(inputs)
            y = y + linear_out
        return tf.nn.sigmoid(y)
