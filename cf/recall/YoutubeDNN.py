import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers import crossnet, mlp, gate, moe
from cf.utils import tensor
from cf.models.base import get_embedding, form_x, model_summary
from cf.preprocess.feature_column import SparseFeat, SequenceFeat
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip
from tensorflow import keras


class YoutubeDNN(Model):
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        """Youtube DNN model

        :param feature_columns: the features used by the model
        :param config:
        :param directory:
        :param args:
        :param kwargs:
        """
        # init model params
        raise NotImplementedError
        super().__init__(*args, **kwargs)
        self.feature_columns = feature_columns
        train_cfg = config['train']
        model_cfg = config['model']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.sparse_len = len(
            list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, SequenceFeat), feature_columns)))
        initializer = keras.initializers.he_normal
        # todo experiment Cowclip in recall model

        # over
        x_dim = self.sparse_len * self.embedding_dim + len(feature_columns) - self.sparse_len
        # model layers
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.dnn = mlp.MLP(model_cfg['dnn_units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                           False, initializer)

        # if len(item_feature_column) > 1:
        #     e = "Now YoutubeNN only support 1 item feature like item_id"
        #     logger.error(e)
        #     raise ValueError(e)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        x = form_x(inputs, self.ebd, False, False)
        x = tensor.to2DTensor(x)
        dnn_out = self.dnn(x)
        dnn_out = tf.nn.l2_normalize(dnn_out, axis=-1)


