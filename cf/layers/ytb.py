import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense


class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        """Embedding index(这是用来干啥的？)

        :param index:
        :param kwargs:
        """
        self.index = index
        super().__init__(**kwargs)
        raise NotImplementedError

    # def build(self, input_shape):
    #     super(EmbeddingIndex, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return tf.constant(self.index)


class L2Norm(Layer):
    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask
