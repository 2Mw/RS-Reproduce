import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras.regularizers import l2
from cf.utils.tensor import to2DTensor


class EmbeddingGate(Layer):
    def __init__(self, num_field, dim, activation='sigmoid', **kwargs):
        """
        The embedding gate in GateNet. For one feature `x`, the output if `activation(W*x) * x`

        :param num_field: The count of fields.
        :param dim: The embedding dim of field.
        :param activation: The activation of embedding gate.
        :param kwargs:
        """
        super(EmbeddingGate, self).__init__(**kwargs)
        self.num_field = num_field
        self.dim = dim
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight('EmbedGateWeight', shape=[self.num_field, self.dim], initializer='random_normal',
                                 regularizer=l2())
        super(EmbeddingGate, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = tf.stack(tf.split(x, self.dim, axis=-1), axis=-1)  # (Batch, F, dim)
        score = tf.einsum('ij,bij->bi', self.W, x)  # (Batch, F)
        # assert score == tf.einsum('ij,bij->bi', self.W, x)
        score = tf.expand_dims(score, -1)
        return to2DTensor(tf.multiply(x, score))


class HiddenGate(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError

    def build(self, input_shape):
        super(HiddenGate, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pass
