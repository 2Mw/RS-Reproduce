import pickle
import warnings

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization, AveragePooling1D
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding
from tensorflow import keras
from cf.layers.mask import MaskedEmbeddingsAggregator as MEA
from cf.utils.logger import logger
from cf.utils.tensor import *
import os


class DoubleTower(Model):
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        super().__init__(*args)
        self.feature_columns = feature_columns
        model_cfg = config['model']
        self.directory = directory
        self.embedding_dim = model_cfg['embedding_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim, mask_zero=True)
        self.temperature = model_cfg['temperature']
        self.avg_pool = AveragePooling1D()
        self.query_mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                                 initializer=keras.initializers.he_normal)
        self.item_mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'],
                                initializer=keras.initializers.he_normal)
        # self.l2 = L2Norm()
        self.bn = BatchNormalization()
        # get the columns information about query and item tower
        if config.get('dataset') is not None:
            cif = config['files'].get(f'{config.get("dataset")}_columns')
            if cif is not None:
                self.query_cols = cif['query']
                self.item_cols = cif['item']
                self.query_id_col = cif['query_id']
                self.item_id_col = cif['item_id']
                self.topk_cmp_col = cif['target_id']
            else:
                e = f'{config.get("dataset")}_columns is not set in `cf.config.dataset`'
                logger.error(e)
                raise ValueError(e)
        else:
            e = f'You must set `dataset` parameter in order to get columns information about query and item tower'
            logger.error(e)
            raise ValueError(e)
        # 保存 query 和 item 的变量
        self.item_data = {}
        self.item_index = None  # 对应索引

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            f.name: Input(shape=(None,), name=f.name)
            for f in self.feature_columns
        }
        model = Model(inputs, outputs=self.call(inputs, training=True))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def train_step(self, data):
        # self-define train_step_demo: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        with tf.GradientTape() as tape:
            loss = self(data[0], training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {}

    def test_step(self, data):
        self(data[0], training=False)
        return {}

    def call(self, inputs, training=None, mask=None):
        # 使用双塔就需要区分 query 字段和 item 字段
        query_x = {k: inputs[k] for k in self.query_cols}
        query_out = self.query_tower(query_x)
        item_x = {k: inputs[k] for k in self.item_cols}
        item_out = self.item_tower(item_x)
        if training:
            # 训练过程，返回 loss
            # 相似度计算最终使用 temperature
            sim = tf.nn.softmax((query_out @ tf.transpose(item_out)) / self.temperature, axis=-1)
            loss = tf.reduce_mean(tf.linalg.diag_part(-tf.math.log(sim)))
            # 保存数据
            self.add_loss(lambda: loss)
            self.add_metric(loss, 'loss')
            return loss
        else:
            # predict / eval 过程，返回对应的 user 向量或者 query 向量
            return query_out, item_out

    def query_tower(self, x):
        x = self.form_x(x)
        x = self.query_mlp(x)
        return tf.math.l2_normalize(x, axis=-1)

    def item_tower(self, x):
        x = self.form_x(x)
        x = self.item_mlp(x)
        return tf.math.l2_normalize(x, axis=-1)

    def form_x(self, x):
        # x - (batch, dim)
        sparse_x = []
        dense_x = []
        seq_x = []
        #
        for f, v in x.items():
            if len(v.shape) == 1:
                v = tf.expand_dims(v, -1)
            key = f
            if ']]' in f:  # Get the key
                key = f.split(']]')[1]
            if f[0] == 'C':
                sparse_x.append(self.ebd[key](v))
            elif f[0] == 'I':
                dense_x.append(tf.expand_dims(v, -1))
            elif f[0] == 'S':
                if tf.__version__ < '2.4.0':
                    mask = tf.expand_dims(tf.cast(self.ebd[key].compute_mask(v), tf.float32), -1)
                    cnt = tf.reduce_sum(mask, 1)
                    masked = tf.multiply(self.ebd[key](v), mask)
                    mid = tf.reduce_sum(masked, axis=1) / cnt
                else:
                    mid = tf.reduce_mean(tf.ragged.boolean_mask(self.ebd[key](v), self.ebd[key].compute_mask(v)), 1)
                mid = tf.math.l2_normalize(mid, 1)
                seq_x.append(tf.expand_dims(mid, 1))
            else:
                warnings.warn(f'The feature:{f} may not be categorized', SyntaxWarning)
        return to2DTensor(tf.concat(sparse_x + dense_x + seq_x, axis=-1))
