import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.regularizers import l2
from cf.layers import crossnet, mlp


class DCN(Model):
    def __init__(self, feature_columns, config, *args, **kwargs):
        """
        Deep & Cross Network

        :param feature_columns:
        :param hidden_units:
        :param activation:
        :param dnn_dropout: The dropout rate.
        :param embedding_reg:
        :param cross_w_reg:
        :param cross_b_reg:
        :param args:
        :param kwargs:
        """
        super(DCN, self).__init__(*args, **kwargs)
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
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

    def call(self, inputs, training=None, mask=None):
        # todo 存在一个问题，所有的 dense 和 sparse feature 全变成了 embedding了
        sparse_embedding = tf.concat([
            self.embedding_layers[feature_name](value)
            for feature_name, value in inputs.items()
        ], axis=1)

        x = sparse_embedding
        # Cross Network
        cross_x = self.cross_net(x)
        # DNN
        dnn_x = self.mlp(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        # TODO 未加正则化
        return tf.nn.sigmoid(self.dense_final(total_x))