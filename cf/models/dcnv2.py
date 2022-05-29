import imp
import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.regularizers import l2
from cf.layers import crossnet, mlp
from cf.utils import tensor


class DCNv2(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        super(DCNv2, self).__init__(*args, **kwargs)
        self.directory = directory
        model_cfg = config['model']
        self.feature_column = feature_column
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'],
                           model_cfg['use_bn'], initializer=keras.initializers.he_normal)
        self.cross = crossnet.CrossNetMix(model_cfg['low_rank'], model_cfg['num_experts'], model_cfg['cross_layers'],
                                          model_cfg['l2_reg_cross'], initializer=keras.initializers.he_normal)

        self.ebd = {
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
            f['name']: Input(shape=(), dtype=tf.float32, name=f['name'])
            for f in self.feature_column
        }
        model = Model(inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, "model.png"), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        # 对于 类别型数据使用 embedding，对于数值型数值不使用 embedding
        x = tf.concat([self.ebd[f](v) if f[0] == 'C' else tf.expand_dims(v, 1) for f, v in inputs.items()], axis=-1)
        x = tensor.to2DTensor(x)
        cross_out = self.cross(x)
        dnn_out = self.mlp(x)
        total_x = tf.concat([cross_out, dnn_out], axis=-1)
        return tf.nn.sigmoid(self.final_dense(total_x))
