import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Layer


class FMLayer(Layer):
    """
    New FM Layer for DeepFM.

    在 DeepFM 中的 FM 层中，只包含一阶和二阶特征信息，不包含 bias 信息
    """

    def __init__(self, feature_length, w_reg=1e-6):
        """
        Init method.

        :param feature_length: A scalar, the length of features.
        :param w_reg: A scalar, the regularization coefficient for weight w.
        """
        super(FMLayer, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=keras.regularizers.l2(self.w_reg),
                                 trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embedding_inputs']
        sparse_inputs = tf.concat([v for _, v in sparse_inputs.items()], axis=1)
        # 一阶计算
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)
        # 二阶计算
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # [batch_size, 1, embed_dim]
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # [batch_size, 1, embed_dim]
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # [batch_size, 1]
        return first_order + second_order
