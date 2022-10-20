import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense


class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        """Embedding index(TODO 这是用来干啥的？)

        :param index:
        :param kwargs:
        """
        self.index = index
        super().__init__(**kwargs)

    # def build(self, input_shape):
    #     super(EmbeddingIndex, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return tf.constant(self.index)
