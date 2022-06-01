import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from cf.layers.attention import MultiheadAttention, AggregationAttention
from cf.layers.mlp import MLP
import os
from keras.layers import Input, Embedding, Conv1D, Dense
from cf.models.base import *


class InterHAt(Model):
    def __init__(self, feature_column, config, directory: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_column = feature_column
        self.directory = directory
        model_cfg = config['model']
        self.embedding_dim = model_cfg['embedding_dim']
        dk = model_cfg['att_dk']
        head_num = model_cfg['att_head_num']
        self.attention = MultiheadAttention(dk, head_num, model_cfg['att_dropout'])
        self.trim_dense = Dense(dk, 'relu')
        self.ff = MLP([4 * dk, dk], model_cfg['activation'], model_cfg['dropout'],
                      model_cfg['use_bn'], model_cfg['use_residual'])
        agg_order = model_cfg['agg_order']
        self.agg = [AggregationAttention(dk, model_cfg['regularization']) for _ in range(agg_order)]
        self.pool = Conv1D(model_cfg['agg_filters'], 1)
        self.weighted_dense = Dense(1, None, False)
        self.final_dense = [
            Dense(self.embedding_dim // 2, model_cfg['activation'], use_bias=False),
            Dense(1, use_bias=False)
        ]

        self.numeric_same = model_cfg['numeric_same_dim']
        self.ebd = get_embedding(feature_column, self.embedding_dim, self.numeric_same, model_cfg['embedding_device'])

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # get embedding
        embedding = form_x(inputs, self.ebd, self.numeric_same)
        # 对于注意力机制层需要将shape修改为 (batch, future, embedding)
        x = tf.reshape(embedding, [-1, len(self.feature_column), self.embedding_dim])

        x = self.attention(x)  # (batch, F, dk*head)

        x = self.trim_dense(x)  # (batch, F, dk)

        x = self.ff(x)  # (batch, F, dk)

        # store nth order information.
        agg_res = []
        x_i = None
        for i, agg in enumerate(self.agg):
            if i > 0:
                t = tf.tile(tf.expand_dims(agg_res[-1], axis=1), [1, x.shape[1], 1])
                x_i = tf.multiply(x, t) + x_i
            else:
                x_i = x
            u, _ = agg(x_i)
            agg_res.append(u)

        all_feature = tf.stack(agg_res, axis=1)  # (Batch, agg_order, dk)
        mapped_feature = self.pool(all_feature)

        # context vector
        weights = tf.nn.softmax(self.weighted_dense(mapped_feature))    # (Batch, agg_order)

        # weighted sum
        weighted_sum_feature = tf.reduce_sum(tf.multiply(all_feature, weights), axis=1)  # (agg_order)

        hidden_logits = self.final_dense[0](weighted_sum_feature)

        # logits = tf.squeeze(self.final_dense[1](hidden_logits), axis=1)
        logits = self.final_dense[1](hidden_logits)  # (batch,)

        # shapes = 1
        # for i in logits.shape[1:]:
        #     if i is not None:
        #         shapes *= i
        # logits = tf.reshape(logits, [-1, shapes])

        return tf.nn.sigmoid(logits)
