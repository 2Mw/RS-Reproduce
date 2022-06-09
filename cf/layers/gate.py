import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, BatchNormalization
from keras.regularizers import l2
from cf.utils.tensor import to2DTensor
from cf.utils.logger import logger


# ============================  1. GateNet part =========================================

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
    """
    The hidden gate in GateNet, NotImplemented
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError

    def build(self, input_shape):
        super(HiddenGate, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pass


# ============================  2. EDCN part =========================================
class BridgeModule(Layer):
    def __init__(self, hidden_dim, bridge_type='hadamard_product', **kwargs):
        """
        The bridge module in EDCN.

        :param hidden_dim: The hidden dim of bridge module, which is used for attention_pooling.
        :param bridge_type: The type of bridge module, optional .
        :param kwargs:
        """
        super().__init__(**kwargs)
        supported_bridge_type = ['hadamard_product', 'pointwise_addition', 'concat', 'attention_pooling']
        if bridge_type not in supported_bridge_type:
            raise ValueError('The bridge_type must be one of {}'.format(supported_bridge_type))
        self.hidden_dim = hidden_dim
        self.bridge_type = bridge_type
        if bridge_type != 'hadamard_product':
            raise NotImplementedError('The bridge_type {} is not implemented yet'.format(bridge_type))

    def call(self, inputs: list, *args, **kwargs):
        if len(inputs) != 2:
            e = f'The input of bridge module must be a list of two tensors, such as [cross_x, dnn_x]'
            logger.error(e)
            raise ValueError(e)
        ax, bx = inputs
        if self.bridge_type == 'hadamard_product':
            # 但是这需要两个维度保持一致才可以
            return tf.multiply(ax, bx)
        else:
            raise NotImplementedError


class RegulationModule(Layer):
    def __init__(self, num_fields, dim, sparse_len, tau=1, use_bn=True, use_regulation=True, **kwargs):
        """
        The regulation module in EDCN.  Regulation module is not only used to regulate the embedding of each field,
        but also used to regulate output of cross layer and hidden layer.

        :param num_fields: The count of fields.
        :param dim: The embedding dim of field.
        :param tau: The tau of gating units.
        :param use_bn: Whether to use batch normalization.
        :param use_regulation: Whether to use regulation.
        :param kwargs:
        :return two gate of pertinent input
        """
        super(RegulationModule, self).__init__(**kwargs)
        self.use_regulation = use_regulation
        if use_regulation:
            self.tau = tau
            self.num_fields = num_fields
            self.dim = dim
            self.use_bn = use_bn
            self.sparse_len = sparse_len
            initializer = keras.initializers.he_normal
            self.g1 = self.add_weight('g1', shape=[self.num_fields], initializer=initializer)
            self.g2 = self.add_weight('g2', shape=[self.num_fields], initializer=initializer)
            if use_bn:
                self.bn1 = BatchNormalization()
                self.bn2 = BatchNormalization()

    def build(self, input_shape):
        super(RegulationModule, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        # input dim: (Batch, F * dim) / (Batch, sparse * dim + numeric)
        x = inputs
        if self.use_regulation:
            # g1: (F, dim), 原文中 g_1^b 是scalar, 为了对每个field对应embedding中每个维度进行相乘，使用repeat
            x1 = tf.expand_dims(tf.nn.softmax(self.g1 / self.tau), axis=-1)  # x1 = (F, 1)
            x2 = tf.expand_dims(tf.nn.softmax(self.g2 / self.tau), axis=-1)
            sparse_x1 = tf.repeat(x1[0:self.sparse_len, :], self.dim, axis=-1)  # sparse_x1 = (F, dim)
            sparse_x2 = tf.repeat(x2[0:self.sparse_len, :], self.dim, axis=-1)
            # g1: (1, F * dim)
            g1, g2 = tf.reshape(sparse_x1, [1, -1]), tf.reshape(sparse_x2, [1, -1])
            if self.sparse_len != x1.shape[0]:
                g1 = tf.concat([g1, tf.reshape(x1[self.sparse_len:, :], [1, -1])], axis=-1)
                g2 = tf.concat([g2, tf.reshape(x1[self.sparse_len:, :], [1, -1])], axis=-1)
            out1, out2 = g1 * x, g2 * x
            if self.use_bn:
                out1, out2 = self.bn1(out1), self.bn2(out2)
            return out1, out2
        else:
            return x, x
