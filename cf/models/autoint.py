import os.path

import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dense
from keras.regularizers import l2
from cf.layers.attention import MultiheadAttention


class AutoInt(Model):
    def __init__(self, feature_columns, config, directory: str = "", *args, **kwargs):
        """
        AutoInt model

        :param config: The configuration of the hyper-parameters.
        :param feature_columns: The feature columns.
        :param directory: The output directory.
        """
        super(AutoInt, self).__init__(*args, **kwargs)
        model_cfg = config['model']
        self.embedding_dim = model_cfg['embedding_dim']
        self.directory = directory
        self.feature_column = feature_columns
        self.embedding_layer = {
            f['name']: Embedding(
                input_dim=f['feature_num'],
                input_length=1,
                output_dim=f['dim'],
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(model_cfg['l2_reg_embedding'])
            )
            for f in feature_columns
        }
        self.att_layer = model_cfg['att_layer_num']
        self.attention = [MultiheadAttention(model_cfg['att_size'], model_cfg['att_head_num']) for i in
                          range(self.att_layer)]
        self.final = Dense(1, activation='sigmoid')

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            f['name']: Input(shape=(), dtype=tf.int32, name=f['name'])
            for f in self.feature_column
        }
        model = Model(inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        x = tf.concat([self.embedding_layer[f](v) for f, v in inputs.items()], axis=1)
        # 对于注意力机制层需要将shape修改为 (batch, future, embedding)
        x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
        for att in self.attention:
            x = att(x)
        shapes = 1
        for i in x.shape[1:]:
            if i is not None:
                shapes *= i
        x = tf.reshape(x, [-1, shapes])
        out = self.final(x)
        return out
