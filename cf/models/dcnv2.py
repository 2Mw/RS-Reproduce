import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.regularizers import l2
from cf.layers import crossnet, mlp
from cf.utils.tensor import *


class DCNv2(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        super(DCNv2, self).__init__(*args, **kwargs)
        self.directory = directory
        model_cfg = config['model']
        self.feature_column = feature_column
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'],
                           model_cfg['use_bn'])
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'])

        self.embedding_layer = {
            f['name']: Embedding(
                input_dim=f['feature_num'],
                input_length=1,
                output_dim=f['dim'],
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(model_cfg['l2_reg_embedding'])
            )
            for f in feature_column
        }
        # self.final_dense = Dense(1, model_cfg['activation'])
        self.final_dense = Dense(1, activation=None)

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False):
        inputs = {
            f['name']: Input(shape=(), dtype=tf.int32, name=f['name'])
            for f in self.feature_column
        }
        model = Model(inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, "model.png"), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        sparse_embedding = tf.concat([
            self.embedding_layer[f](v)
            for f, v in inputs.items()
        ], axis=1)

        x = to2DTensor(sparse_embedding)
        cross_out = self.cross(x)
        dnn_out = self.mlp(x)
        total_x = tf.concat([cross_out, dnn_out], axis=-1)
        return tf.nn.sigmoid(self.final_dense(total_x))
