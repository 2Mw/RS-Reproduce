import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers.attention import MultiheadAttention
from cf.models.base import get_embedding, model_summary, form_x
from cf.layers import linear, mlp
from cf.preprocess.feature_column import SparseFeat
from cf.utils.tensor import to2DTensor
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip


class AutoInt(Cowclip):
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
            super(AutoInt, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(AutoInt, self).__init__(*args, **kwargs)
        # att params
        self.att_size = model_cfg['att_size']
        self.head_num = model_cfg['att_head_num']
        self.att_layer = model_cfg['att_layer_num']
        self.units = model_cfg['hidden_units']
        # networks
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.attention = [MultiheadAttention(self.att_size, self.head_num) for i in range(self.att_layer)]
        self.final = Dense(1, use_bias=False)
        self.linear = linear.Linear(feature_columns)
        self.mlp = mlp.MLP(self.units, model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # Attention
        att_x, dense_x = form_x(inputs, self.ebd, True, self.numeric_same_dim)
        # 对于注意力机制层需要将shape修改为 (batch, feature, embedding)
        if self.numeric_same_dim:
            x = tf.concat([att_x, dense_x], axis=-1)
            x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
        else:
            x = tf.reshape(att_x, [-1, self.sparse_len, self.embedding_dim])
        for att in self.attention:
            x = att(x)
        out = to2DTensor(x)
        # dnn if AutoInt+ to model implicit features.
        if len(self.units) > 0:
            dnn_out = self.mlp(tf.concat([att_x, dense_x], axis=-1))
            out = tf.concat([out, dnn_out], axis=-1)
        y = self.final(out)
        if self.linear_res:
            linear_out = self.linear(inputs)
            y = y + linear_out
        return tf.nn.sigmoid(y)
