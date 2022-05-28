import os.path

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.regularizers import l2
from cf.layers import crossnet, mlp


class DCN(Model):
    def __init__(self, feature_columns, config, directory: str = '', *args, **kwargs):
        """
        Deep & Cross Network

        :param feature_columns: The feature columns
        :param cfg: The config of hyper parameters.
        :param directory: The directory of the model.
        :param args:
        :param kwargs:
        """
        super(DCN, self).__init__(*args, **kwargs)
        self.directory = directory
        model_config = config['model']
        self.feature_columns = feature_columns
        self.hidden_units = model_config['hidden_units']
        self.layer_num = len(self.hidden_units)
        self.dnn_dropout = model_config['dropout']
        self.activation = model_config['activation']
        self.embedding_layers = {
            feature['name']: Embedding(
                input_dim=feature['feature_num'],
                input_length=1,
                output_dim=feature['dim'],
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(model_config['embedding_reg'])
            )
            for feature in self.feature_columns
        }
        self.cross_net = crossnet.CrossNet(self.layer_num, model_config['cross_w_reg'], model_config['cross_b_reg'])
        self.mlp = mlp.MLP(self.hidden_units, self.activation, self.dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False):
        inputs = {
            feature['name']: Input(shape=(), dtype=tf.int32, name=feature['name'])
            for feature in self.feature_columns
        }
        model = Model(inputs=inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        # todo 存在一个问题，所有的 dense 和 sparse feature 全变成了 embedding了
        sparse_embedding = tf.concat([
            self.embedding_layers[feature_name](value)
            for feature_name, value in inputs.items()
        ], axis=1)
        print(f'test shape{self.embedding_layers["C1"](1).shape}')
        x = sparse_embedding
        print(f'========== x.shape:  {x.shape} ===========')
        # Cross Network
        cross_x = self.cross_net(x)
        # DNN
        dnn_x = self.mlp(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        # TODO 未加正则化
        return tf.nn.sigmoid(self.dense_final(total_x))
