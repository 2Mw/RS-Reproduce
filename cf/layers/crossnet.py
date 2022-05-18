import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Layer
from keras.layers import Dense
from keras.regularizers import l2


class CrossNet(Layer):
    def __init__(self, layer_num, reg_w=0., reg_b=0., **kwargs):
        """
        CrossNetwork for Deep & Cross

        :param layer_num: The depth of layers.
        :param reg_w: The regularization coefficient of weights.
        :param reg_b: The regularization coefficient of biases.
        :param kwargs: Other parameters.
        """
        super().__init__(**kwargs)
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name=f'w_{i}',
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]

        self.cross_bias = [
            self.add_weight(name=f'b_{i}',
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, *args, **kwargs):
        # input dim: (Batch, dim)
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch, dim, 1)
        x_l = x_0
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch, 1, 1)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch, dim)
        return x_l


class CrossNetMix(Layer):
    """
    The cross network part of DCN-V2 model, which improves DCN by:

    -  Add MOE to learn feature interactions in different subspaces.

    -  Add nonlinear transformations in low-dimensional space

    Input shape
    - 2D tensor with shape: ``(batch_size, units)``.

    Output shape
    - 2D tensor with shape: ``(batch_size, units)``.


    Initialize arguments for CrossNetMix.

     - low_rank: Positive integer, the dimensionality of low-rank space.
     - num_experts: Positive integer, the number of experts.
     - layer_num: Positive interger, the cross layer number.
     - l2_reg: float between 0 and 1. L2 regularization strength applied to the kernel weights matrix.
     - seed: A python integer to use as random seed.
    """

    def __init__(self, low_rank: int, num_experts: int = 4, layer_num: int = 2, l2_reg: float = 0., seed=1024,
                 **kwargs):
        if low_rank <= 0:
            raise ValueError(f'The low_rank parameter must be positive, but yours is {low_rank}')

        if num_experts <= 0:
            raise ValueError(f'The num_experts parameter must be positive, but yours is {num_experts}')

        if layer_num <= 0:
            raise ValueError(f'The num_experts parameter must be positive, but yours is {layer_num}')

        if l2_reg < 0 or l2_reg > 1:
            raise ValueError(f'The l2_reg is illegal, must be in the range of (0,1), but {l2_reg}')

        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f'Unexpected inputs dimensions {len(input_shape)}, expect to be 2 dimensions')

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name=f'U_list_{i}',
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=keras.initializers.glorot_normal(self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]

        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name=f'V_list_{i}',
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=keras.initializers.glorot_normal(self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]

        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name=f'C_list_{i}',
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=keras.initializers.glorot_normal(self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True
                                       ) for i in range(self.layer_num)]

        self.gating = [Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name=f'bias_{i}',
                                     shape=(dim, 1),
                                     initializer=keras.initializers.Zeros(),
                                     trainable=True) for i in range(self.layer_num)]

        super(CrossNetMix, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if keras.backend.ndim(inputs) != 2:
            raise ValueError(f'Unexpected inputs dimensions {keras.backend.ndim(inputs)}, expect to be 2 dimensions')

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score = []
            for eid in range(self.num_experts):
                gating_score.append(self.gating[eid](tf.squeeze(x_l, axis=2)))

                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][eid]), x_l)  # 爱因斯坦求和

                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][eid], v_x)
                v_x = tf.nn.tanh(v_x)

                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][eid], v_x)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_
                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score = tf.stack(gating_score, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score, 1))
            x_l = moe_out + x_l
        x_l = tf.squeeze(x_l, 2)
        return x_l

    def get_config(self):
        config = {
            'low_rank': self.low_rank,
            'num_experts': self.num_experts,
            'layer_num': self.layer_num,
            'l2_reg': self.l2_reg,
            'seed': self.seed
        }
        base_config = super(CrossNetMix, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape
