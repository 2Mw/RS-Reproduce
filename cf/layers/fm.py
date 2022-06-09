import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from cf.utils.logger import logger
from cf.utils.tensor import to2DTensor


class FMLayer(Layer):
    """
    New FM Layer for DeepFM.

    在 DeepFM 中的 FM 层中，只包含一阶和二阶特征信息，不包含 bias 信息
    """

    def __init__(self, w_reg=1e-6):
        """
        Init method.

        :param w_reg: A scalar, the regularization coefficient for weight w.
        """
        super(FMLayer, self).__init__()
        self.w_reg = w_reg
        self.dense = Dense(1)

    def build(self, input_shape):
        # input_shape (batch, F, dim)
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if len(inputs.shape) != 3:
            e = f'The dimension of input in the model DeepFM must be 3.'
            logger.error(e)
            raise ValueError(e)
        x = inputs  # (batch ,F, D)
        # 计算一阶
        x_1 = to2DTensor(x)
        first_order = self.dense(x_1)
        # 计算二阶，这里在 DeepFM 中不需要考虑权重 V，因为在 embedding layer 中已经参与计算，见 DeepFM figure 4.
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(x * x, axis=1, keepdims=True)
        second_order = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)
        # print(second_order.numpy())
        return first_order + second_order
