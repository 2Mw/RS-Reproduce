import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from cf.layers import mlp

"""Mixture of Experts"""


class MMOE(Layer):
    def __init__(self, num_experts, expert_dnn_units, num_gates, gate_dnn_units, activation='relu', dropout=0.,
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
