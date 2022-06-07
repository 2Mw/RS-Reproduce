import tensorflow as tf
from keras.models import Model
from cf.layers.attention import MultiheadAttention, AggregationAttention
from cf.layers import gate, mlp
from keras.layers import Conv1D, Dense
from cf.models.base import model_summary, form_x, get_embedding
from cf.preprocess.feature_column import SparseFeat
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip


class InterHAt(Cowclip):
    def __init__(self, feature_column, config, directory: str = '', *args, **kwargs):
        # model params
        self.feature_column = feature_column
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.embedding_dim = model_cfg['embedding_dim']
        dk = model_cfg['att_dk']
        head_num = model_cfg['att_head_num']
        agg_order = model_cfg['agg_order']
        self.sparse_len = len(list(filter(lambda x: isinstance(x, SparseFeat), feature_column)))
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(InterHAt, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(InterHAt, self).__init__(*args, **kwargs)
        # Optional params
        self.use_embed_gate = model_cfg['use_embed_gate']
        # model layers
        self.attention = MultiheadAttention(dk, head_num, model_cfg['att_dropout'])
        self.ff = mlp.MLP([4 * dk, dk], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                          model_cfg['use_residual'])
        self.agg = [AggregationAttention(dk, model_cfg['regularization']) for _ in range(agg_order)]
        self.pool = Conv1D(model_cfg['agg_filters'], 1)
        self.weighted_dense = Dense(1, None, False)
        self.trim_dense = Dense(dk, 'relu')
        self.final_dense = [
            Dense(self.embedding_dim // 2, model_cfg['activation'], use_bias=False),
            Dense(1, use_bias=False)
        ]
        self.ebd = get_embedding(feature_column, self.embedding_dim, model_cfg['embedding_device'])
        if self.use_embed_gate:
            self.embed_gate = gate.EmbeddingGate(self.sparse_len, self.embedding_dim)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # get embedding
        sparse_x, dense_x = form_x(inputs, self.ebd, True, self.numeric_same_dim)
        # Embedding Gate
        if self.use_embed_gate:
            sparse_x = self.embed_gate(sparse_x)
        # 对于注意力机制层需要将shape修改为 (batch, future, embedding)
        if self.numeric_same_dim:
            x = tf.concat([sparse_x, dense_x], axis=-1)
            x = tf.reshape(x, [-1, len(self.feature_column), self.embedding_dim])
        else:
            x = tf.reshape(sparse_x, [-1, self.sparse_len, self.embedding_dim])

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
        weights = tf.nn.softmax(self.weighted_dense(mapped_feature))  # (Batch, agg_order)

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
