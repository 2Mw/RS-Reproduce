from keras.layers import Dense, Dropout, BatchNormalization, Layer
from tensorflow import keras
from cf.layers.activation import PReLU
import tensorflow as tf


class MLP(Layer):
    def __init__(self, units, activation: str, dropout: float, use_bn: bool = False, residual: bool = False,
                 initializer=None, **kwargs):
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
        self.initializer = initializer if initializer is not None else keras.initializers.glorot_normal
        self.activation = activation
        if activation is not None and activation.lower() == 'prelu' and tf.__version__ < '2.4.0':
            self.activation_fn = PReLU()
        else:
            self.activation_fn = activation
        self.dnn = [Dense(unit, self.activation_fn) for unit in units]
        self.dropout = Dropout(dropout)
        self.bn = [BatchNormalization() for unit in units]
        self.use_bn = use_bn
        self.residual = residual

    def call(self, inputs, *args, **kwargs):
        # 2sAGCN中是 bn->relu->dropout
        x = inputs
        for i, dense in enumerate(self.dnn):
            x = dense(x)
            if self.activation.lower() == 'prelu' and tf.__version__ < '2.4.0':
                x = self.activation_fn(x)
            if self.use_bn:
                x = self.bn[i](x)
            # dropout 要放在 bn 层后面
            x = self.dropout(x)
        if self.residual:
            x = x + inputs
        return x
