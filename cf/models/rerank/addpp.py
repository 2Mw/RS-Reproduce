import pickle
import warnings

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization, AveragePooling1D
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding
from tensorflow import keras
from cf.layers import moe, gate
from cf.utils.logger import logger
from cf.utils.tensor import *
from cf.preprocess.feature_column import SparseFeat, SequenceFeat
import os
import seaborn as sns
import time
import math

class ADDPP(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args)
        model_cfg = config['model']
        self.num_experts = model_cfg['mmoe_experts']
        self.units = model_cfg['units']
        # self.dnn_mmoe = moe.MMOE(self.num_experts, self.units, 2, dropout=model_cfg['dropout'])
        self.dnn_mmoe = moe.MMoE_(4, 2, self.units, 'PReLU', model_cfg['dropout'])
        
    def train_step(self, data):
        # self-define train_step_demo: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # y_pred = tf.transpose(tf.reshape(y_pred, [2, -1]))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, mask=None):
        weights = self.dnn_mmoe(inputs)
        return weights
    
"""
Our proposed fast implementation of the greedy algorithm
:param kernel_matrix: 2-d array
:param max_length: positive int
:param epsilon: small positive scalar
:return: list
"""
def dpp(kernel_matrix, max_length, relavance_score=None, relavance_weight=0, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        if relavance_score is None:
            selected_item = np.argmax(di2s)
        else:
            di2s_cp = np.copy(di2s)
            di2s_cp[di2s_cp == -np.inf] = 0
            di2s_cp = (di2s_cp - np.min(di2s_cp)) / (np.max(di2s_cp) - np.min(di2s_cp))
            ad_dpp_matrix = relavance_weight * relavance_score + (1 - relavance_weight) * di2s_cp
            selected_item = np.argmax(ad_dpp_matrix)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_sw(kernel_matrix, window_size, max_length, relavance_score=None, relavance_weight=0, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        if relavance_score is None:
            selected_item = np.argmax(di2s)
        else:
            di2s_cp = np.copy(di2s)
            di2s_cp[di2s_cp == -np.inf] = 0
            di2s_cp = (di2s_cp - np.min(di2s_cp)) / (np.max(di2s_cp) - np.min(di2s_cp))
            ad_dpp_matrix = relavance_weight * relavance_score + (1 - relavance_weight) * di2s_cp
            selected_item = np.argmax(ad_dpp_matrix)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


'''
Maximum Marginal Relevance
'''
def mmr(sim_matrix, max_length, relavance_score, relavance_weight=0.5):
    s = []
    u = [i for i in range(sim_matrix.shape[0])]
    idx = np.argmax(relavance_score)
    s.append(idx)
    u.remove(idx)
    
    sim_matrix = sim_matrix - np.diag(np.diag(sim_matrix))
    
    for _ in range(max_length - 1):
        max_mr_i, max_i = -0x3f3f3f3f, 0
        for i in u:
            mr_i = relavance_weight * relavance_score[i] - (1-relavance_weight) * np.max(sim_matrix[i][s])
            if mr_i > max_mr_i:
                max_mr_i = mr_i
                max_i = i
        s.append(max_i)
        u.remove(max_i)
    
    return s