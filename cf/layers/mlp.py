import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Layer
from keras.layers import Dense, Dropout, BatchNormalization


class MLP(Layer):
    def __init__(self, units, activation: str, dropout: float, use_bn: bool = False, residual: bool = False, **kwargs):
        """
        Multi-Layer perceptron.

        :param units: The output units of per layer.
        :param activation: activation function.
        :param dropout: Dropout rate
        :param use_bn: bool. if use batch normalization.
        :param residual: bool. If use residual connection.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dnn = [Dense(unit, activation) for unit in units]
        self.dropout = Dropout(dropout)
        self.bn = BatchNormalization()
        self.use_bn = use_bn
        self.residual = residual

    def call(self, inputs, *args, **kwargs):
        # 2sAGCN中是 bn->relu->dropout
        x = inputs
        for dense in self.dnn:
            x = dense(x)
            if self.use_bn:
                x = self.bn(x)
            # dropout 要放在 bn 层后面
            x = self.dropout(x)
        if self.residual:
            x = x + inputs
        return x
