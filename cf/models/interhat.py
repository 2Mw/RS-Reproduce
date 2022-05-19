import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from cf.layers.attention import MultiheadAttention, AggregationAttention
from cf.layers.mlp import MLP
import os
from keras.layers import Input, Embedding, Conv1D, Dense
from keras.regularizers import l2


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

        self.embedding_layer = {
            f['name']: Embedding(
                input_dim=f['feature_num'],
                input_length=1,
                output_dim=f['dim'],
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(model_cfg['l2_reg_embedding'])
            )
            for f in feature_column
        }

    def build(self, input_shape):
        super(InterHAt, self).build(input_shape)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            feature['name']: Input(shape=(), dtype=tf.int32, name=feature['name'])
            for feature in self.feature_column
        }
        model = Model(inputs=inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        # get embedding
        embedding = tf.concat([self.embedding_layer[f](v) for f, v in inputs.items()], axis=1)
        # 对于注意力机制层需要将shape修改为 (batch, future, embedding)
        x = tf.reshape(embedding, [-1, len(self.feature_column), self.embedding_dim])

        x = x
        x = self.attention(x)

        x = self.trim_dense(x)  # (batch, F, dim)

        x = self.ff(x)  # (batch, F, dim)

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

        all_feature = tf.stack(agg_res, axis=1)
        mapped_feature = self.pool(all_feature)

        # context vector
        weights = tf.nn.softmax(tf.squeeze(self.weighted_dense(mapped_feature), axis=2))

        # weighted sum
        weighted_sum_feature = tf.reduce_sum(tf.multiply(all_feature, tf.expand_dims(weights, axis=2)), axis=1)

        hidden_logits = self.final_dense[0](weighted_sum_feature)

        # logits = tf.squeeze(self.final_dense[1](hidden_logits), axis=1)
        logits = self.final_dense[1](hidden_logits)

        # shapes = 1
        # for i in logits.shape[1:]:
        #     if i is not None:
        #         shapes *= i
        # logits = tf.reshape(logits, [-1, shapes])

        return tf.nn.sigmoid(logits)
