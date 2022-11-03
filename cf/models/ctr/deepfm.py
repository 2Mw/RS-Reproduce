import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from cf.layers import fm, mlp, linear
from cf.models.ctr.base import get_embedding, model_summary, form_x
from cf.models.ctr.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class DeepFM(Model):
    def __init__(self, feature_columns, config, directory: str = '', *args, **kwargs):
        # TODO 需要修复
        """

        :param feature_columns:  A list. [{'name':, 'vocab_size':, 'dim':}, ...]
        :param config:  Hyper parameters configurations.
        :param directory: The directory of the model.
        """
        # model params
        self.directory = directory
        self.feature_column = feature_columns
        self.config = config
        model_cfg = config['model']
        train_cfg = config['train']
        self.embedding_dim = model_cfg['embedding_dim']
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)))
        self.linear_res = model_cfg['linear_res']
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(DeepFM, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DeepFM, self).__init__(*args, **kwargs)
        # Layer initialization
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.fm = fm.FMLayer(model_cfg['fm_w_reg'])
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'])
        self.dense = keras.layers.Dense(units=1, activation=None)
        self.linear = linear.Linear(feature_columns)

    def call(self, inputs, **kwargs):
        # embedding, (batch_size, embedding_dim*fields)
        sparse_x, dense_x = form_x(inputs, self.ebd, True)
        # wide
        wide_input = tf.reshape(sparse_x, [-1, self.sparse_len, self.embedding_dim])
        wide_outputs = self.fm(wide_input)
        # deep
        dnn_x = tf.concat([sparse_x, dense_x], axis=-1)
        deep_outputs = self.mlp(dnn_x)
        deep_outputs = self.dense(deep_outputs)
        out = tf.add(wide_outputs, deep_outputs)
        if self.linear_res:
            linear_out = self.linear(inputs)
            out = out + linear_out
        return tf.nn.sigmoid(out)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)
