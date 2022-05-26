import copy

import tensorflow as tf
from keras.api._v2 import keras
from keras.layers import BatchNormalization, Dropout, Dense, Layer
from keras.regularizers import L2


class MultiheadAttention(Layer):
    """
    A Layer used in AutoInt that the model the correlation between different feature fields by multi-head
    self-attention mechanism.

    Input shape:
        - A 3D tensor with shape ``(batch_size, field_size, embedding_size)``

    Output shape:
        - 3D tensor with shape: ``(batch_size, field_size, att_embedding_size * head_num)``

    Arguments:
         - dk: int. d_model = d_k * head_num
         - head_num: int. The head number in multi-head self-attention network.
         - dropout: float. The dropout rate of scale dot product attention
         - seed: random seed.
    """

    def __init__(self, dk: int, head_num: int, dropout: float = 0, seed=1024, **kwargs):
        super().__init__(**kwargs)
        self.dropout = Dropout(dropout)
        self.dk = dk
        self.head_num = head_num
        self.seed = seed

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

        embedding_size = int(input_shape[-1])
        self.Q = self.add_weight(name='query', shape=[embedding_size, self.dk * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed))
        self.K = self.add_weight(name='key', shape=[embedding_size, self.dk * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.V = self.add_weight(name='value', shape=[embedding_size, self.dk * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed + 2))

        self.res = self.add_weight(name="res", shape=[embedding_size, self.dk * self.head_num],
                                   dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed))

        self.bn = BatchNormalization()

        super(MultiheadAttention, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if keras.backend.ndim(inputs) != 3:  # (batch, F, dim)
            raise ValueError(f"Unexpected inputs dimensions {keras.backend.ndim(inputs)}, expect to be 3 dimensions")

        querys = tf.tensordot(inputs, self.Q, axes=(-1, 0))  # (batch, F, dim * head_num)
        keys = tf.tensordot(inputs, self.K, axes=(-1, 0))
        values = tf.tensordot(inputs, self.V, axes=(-1, 0))

        # 拆分
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # (head_num*batch, F, dim)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        # scaled dot product attention
        weight = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # (head_num*batch, F, F)
        # inner_product = tf.matmul(querys, keys, transpose_b=True)  # transpose_b b在乘法前进行转置

        weight /= self.dk ** 0.5

        att_scores = tf.nn.softmax(weight, axis=-1)  # (head_num*batch, F, F)
        att_scores = self.dropout(att_scores)
        result = tf.matmul(att_scores, values)  # (head_num*batch, F, dim)

        result = tf.concat(tf.split(result, self.head_num), axis=-1)  # (batch, F, dim*head_num)
        # result = tf.squeeze(result, axis=0)

        # residual connection
        result += tf.tensordot(inputs, self.res, axes=(-1, 0))

        # BatchNormalization
        result = self.bn(result)

        return result

    def compute_output_shape(self, input_shape):
        return (None, input_shape, self.dk * self.head_num)

    def get_config(self):
        cfg = {
            'att_embedding_size': self.dk,
            'head_num': self.head_num,
            'use_res': self.use_res,
            'seed': self.seed
        }

        base_cfg = super(MultiheadAttention, self).get_config()
        base_cfg.update(cfg)
        return base_cfg


class AggregationAttention(Layer):
    def __init__(self, dk, regularize_scale: float, **kwargs):
        super().__init__(**kwargs)
        self.dk = dk
        regularizer = L2(regularize_scale)
        self.project = Dense(self.dk, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer)

    def build(self, input_shape):
        self.query = self.add_weight('query', [self.dk, ], dtype=tf.float32)
        super(AggregationAttention, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        keys = value = inputs
        projected = self.project(keys)
        query = tf.reshape(self.query, [1, 1, -1])

        attention_energy = tf.reduce_sum(tf.multiply(projected, query), axis=2)
        attentions = tf.nn.softmax(attention_energy)
        results = tf.reduce_sum(tf.multiply(value, tf.expand_dims(attentions, axis=2)), axis=1)
        return results, attentions
