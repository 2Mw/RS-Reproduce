import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from cf.layers import mlp
from keras.layers import Dense, Dropout, BatchNormalization, Layer
from tensorflow import keras
from cf.layers.activation import PReLU
import tensorflow as tf

"""Mixture of Experts"""


class MMOE(Layer):
    def __init__(self, num_experts, expert_dnn_units, num_gates, gate_dnn_units=[], activation=None, dropout=0.,
                 use_bn=False, residual=False, initializer=None, **kwargs):
        """
        Multi-Mixture of Experts model, return the gates(task) numbers output.

        :param num_experts: int, number of experts
        :param expert_dnn_units: list, list of positive integer, the layer number and units in each layer of expert DNN
        :param num_gates: int, number of gates
        :param gate_dnn_units: list, list of positive integer, the layer number and units in each layer of gate DNN
        :param activation: str, activation function to use in DNN
        :param dropout: float in [0,1), the probability we will drop out a given DNN coordinate
        :param use_bn: bool, whether use BatchNormalization before activation or not in DNN
        :param residual: bool, whether use residual connection or not
        :param kwargs:
        """
        super().__init__(**kwargs)
        initializer = keras.initializers.glorot_normal if initializer is None else initializer
        self.experts = [mlp.MLP(expert_dnn_units, activation, dropout, use_bn, residual, initializer) for _ in
                        range(num_experts)]
        self.gates = [{
            'input': mlp.MLP(gate_dnn_units, activation, dropout, use_bn, residual, initializer),
            'output': Dense(num_experts, 'softmax', False, name=f'gate_softmax_{i}')
        } for i in range(num_gates)]

    def call(self, inputs, *args, **kwargs):
        x = inputs  # (Batch, dim)
        experts_out = []
        for e in self.experts:
            experts_out.append(e(x))  # e(x): (Batch, units[-1])
        experts_out = tf.stack(experts_out, axis=-1)  # (Batch, units[-1], num_experts)
        mmoe_out = []
        for gate in self.gates:
            gate_out = tf.expand_dims(gate['output'](gate['input'](x)), -1)  # (Batch, num_experts, 1)
            # mmoe_out.append(tf.reduce_sum(experts_out * gate_out, axis=1))  # (Batch, units[-1])
            mmoe_out.append(tf.squeeze(experts_out @ gate_out, axis=-1))  # item: (Batch, units[-1])
        return mmoe_out


"""_summary_
Multi-Gate Mixture of Expert
"""
class MMoE_(Layer):
    def __init__(self, num_experts, number_gates, units, activation: str, dropout: float, use_bn: bool = False, residual: bool = False,
                 initializer=None, **kwargs):
        """
        Multi-Mixture of Experts model, return the gates(task) numbers output.

        :param num_experts: int, number of experts
        :param number_gates: int, number of gates
        :param units: The output units of per layer.
        :param activation: activation function.
        :param dropout: Dropout rate
        :param use_bn: bool. if use batch normalization.
        :param residual: bool. If use residual connection.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.number_gates = number_gates
        self.initializer = initializer if initializer is not None else keras.initializers.glorot_normal
        self.activation = activation
        if activation is not None and activation.lower() == 'prelu' and tf.__version__ < '2.4.0':
            self.activation_fn = None
        else:
            self.activation_fn = activation
        self.dnn = [Dense(unit, self.activation_fn) for unit in units]
        if activation is not None and activation.lower() == 'prelu' and tf.__version__ < '2.4.0':
            self.activation_fn = [PReLU() for unit in units]
        self.dropout = Dropout(dropout)
        self.bn = [BatchNormalization() for unit in units]
        self.use_bn = use_bn
        self.residual = residual

    def call(self, inputs, *args, **kwargs):
        # 2sAGCN中是 bn->relu->dropout
        x = inputs
        for i, dense in enumerate(self.dnn):
            x = dense(x)
            if self.activation is not None and self.activation.lower() == 'prelu' and tf.__version__ < '2.4.0':
                x = self.activation_fn[i](x)
            if self.use_bn:
                x = self.bn[i](x)
            # dropout 要放在 bn 层后面
            x = self.dropout(x)
        if self.residual:
            x = x + inputs
        return x