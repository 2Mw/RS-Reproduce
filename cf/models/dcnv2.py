import tensorflow as tf
from tensorflow import keras
from cf.models.cowclip import Cowclip
from keras.models import Model
from keras.layers import Dense
from cf.models.base import get_embedding, form_x, model_summary
from cf.layers import crossnet, mlp, linear
from cf.utils.tensor import to2DTensor
from cf.models.base import checkCowclip


class DCNv2(Cowclip):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        # model params
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.feature_column = feature_column
        self.embedding_dim = model_cfg['embedding_dim']
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(DCNv2, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DCNv2, self).__init__(*args, **kwargs)
        # model layers
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'],
                           model_cfg['use_bn'], initializer=keras.initializers.he_normal)
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'], initializer=keras.initializers.he_normal)
        self.ebd = get_embedding(feature_column, self.embedding_dim, model_cfg['embedding_device'])
        self.linear = linear.Linear(feature_column)
        self.final_dense = Dense(1, activation=None)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # 对于 类别型数据使用 embedding，对于数值型数值不使用 embedding
        linear_out = self.linear(inputs)
        x = form_x(inputs, self.ebd, False)
        x = to2DTensor(x)
        cross_out = self.cross(x)
        dnn_out = self.mlp(x)
        total_x = tf.concat([cross_out, dnn_out], axis=-1)
        y = self.final_dense(total_x) + linear_out
        return tf.nn.sigmoid(y)
