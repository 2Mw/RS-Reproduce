import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Layer


class Interaction(Layer):
    """
    A Layer used in AutoInt that the model the correlation between different feature fields by multi-head
    self-attention mechanism.

    Input shape:
        - A 3D tensor with shape ``(batch_size, field_size, embedding_size)``

    Output shape:
        - 3D tensor with shape: ``(batch_size, field_size, att_embedding_size * head_num)``

    Arguments:
         - att_embedding_size: int. The embedding size in multi-head self-attention network.
         - head_num: int. The head number in multi-head self-attention network.
         - use_res: bool. Whether you use standard residual connections before output.
         - seed: random seed.

    """

    def __init__(self, att_embedding_size: int, head_num: int, use_res: bool, scaling: bool = False, seed=1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

        embedding_size = int(input_shape[-1])
        self.Q = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed))
        self.K = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.V = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                 dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed + 2))

        if self.use_res:
            self.res = self.add_weight(name="res", shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=keras.initializers.TruncatedNormal(seed=self.seed))

        super(Interaction, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if keras.backend.ndim(inputs) != 2:  # (batch, dim)
            raise ValueError(f"Unexpected inputs dimensions {keras.backend.ndim(inputs)}, expect to be 3 dimensions")

        querys = tf.tensordot(inputs, self.Q, axes=(-1, 0))  # (batch, dim * head_num)
        keys = tf.tensordot(inputs, self.K, axes=(-1, 0))
        values = tf.tensordot(inputs, self.V, axes=(-1, 0))

        # 拆分
        querys = tf.stack(tf.split(querys, self.head_num, axis=1))  # (head_num, batch, dim)
        keys = tf.stack(tf.split(keys, self.head_num, axis=1))
        values = tf.stack(tf.split(values, self.head_num, axis=1))

        weight = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # (head_num, batch, batch)
        # inner_product = tf.matmul(querys, keys, transpose_b=True)  # transpose_b b在乘法前进行转置

        if self.scaling:
            weight /= self.att_embedding_size ** 0.5

        self.att_scores = tf.nn.softmax(weight, axis=-1)

        result = tf.matmul(self.att_scores, values)
        result = tf.concat(tf.split(result, self.head_num), axis=-1)
        result = tf.squeeze(result, axis=0)

        if self.use_res:
            result += tf.tensordot(inputs, self.res, axes=(-1, 0))

        result = tf.nn.relu(result)

        # TODO normalize

        return result

    def compute_output_shape(self, input_shape):
        return None, input_shape, self.att_embedding_size * self.head_num

    def get_config(self):
        cfg = {
            'att_embedding_size': self.att_embedding_size,
            'head_num': self.head_num,
            'use_res': self.use_res,
            'seed': self.seed
        }

        base_cfg = super(Interaction, self).get_config()
        base_cfg.update(cfg)
        return base_cfg
