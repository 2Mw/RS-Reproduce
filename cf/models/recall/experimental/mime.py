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


class MIME(Model):
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        super().__init__(*args)
        self.feature_columns = feature_columns
        model_cfg = config['model']
        self.directory = directory
        self.embedding_dim = model_cfg['embedding_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim, mask_zero=True)
        self.temperature = model_cfg['temperature']
        self.avg_pool = AveragePooling1D()
        self.activation = model_cfg['activation']
        self.units = model_cfg['units']
        self.num_interest = model_cfg['interests']
        self.query_mlp = mlp.MLP(self.units, self.activation, model_cfg['dropout'], model_cfg['use_bn'],
                                 initializer=keras.initializers.he_normal)
        self.item_mlp = [
            [
                mlp.MLP([unit], self.activation, model_cfg['dropout'], model_cfg['use_bn'],
                        initializer=keras.initializers.he_normal) for unit in self.units
            ] for _ in range(self.num_interest)
        ]
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
        # multi-interests configuration
        self.item_sparse_len = len(
            list(filter(
                lambda x: x.name in self.item_cols and (isinstance(x, SparseFeat) or isinstance(x, SequenceFeat)),
                feature_columns)))
        x_dim = self.item_sparse_len * self.embedding_dim + len(self.item_cols) - self.item_sparse_len
        self.num_experts = model_cfg['mmoe_experts']
        self.ebd_mmoe = moe.MMOE(self.num_experts, [x_dim], self.num_interest, dropout=model_cfg['dropout'])
        self.dnn_mmoe = [moe.MMOE(self.num_experts, [self.units[0]], 2, dropout=model_cfg['dropout']) for _ in
                         range(self.num_interest - 1)]

        self.dnn_bridge = [gate.BridgeModule(model_cfg['bnn_bridge_type']) for _ in range(self.num_interest - 1)]
        self.use_dnn_mmoe = model_cfg['use_dnn_mmoe']
        self.merge_strategy = model_cfg['merge_strategy']
        self.merge_dnn = mlp.MLP([self.units[-1]], self.activation, model_cfg['dropout'], model_cfg['use_bn'],
                                 initializer=keras.initializers.he_normal)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            f.name: Input(shape=(None,))
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
            loss = tf.reduce_mean(tf.linalg.diag_part((-1) * tf.math.log(sim)))
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
        xs = self.ebd_mmoe(x)
        mlp_out = []
        for i in range(len(self.units)):
            mlp_out = [self.item_mlp[idx][i](_x) for idx, _x in enumerate(xs)]
            if self.use_dnn_mmoe and self.num_interest > 1 and i == 0:
                combined = []
                for j in range(self.num_interest - 1):
                    combined.append(self.dnn_bridge[j]([mlp_out[j], mlp_out[j + 1]]))
                dnn_mmoe_out = []
                for j, item in enumerate(self.dnn_mmoe):
                    dnn_mmoe_out.extend(item(combined[j]))
                res = [dnn_mmoe_out.pop(0)]
                while len(dnn_mmoe_out) > 2:
                    a = dnn_mmoe_out.pop(0)
                    b = dnn_mmoe_out.pop(1)
                    res.append(tf.concat([a, b], axis=-1))
                res = res + dnn_mmoe_out
                mlp_out = res
            xs = mlp_out

        if self.num_interest > 1:
            if self.merge_strategy == 'mean':
                xs = tf.concat([tf.expand_dims(i, axis=-1) for i in xs], axis=-1)
                x = tf.reduce_mean(xs, axis=-1)
            elif self.merge_strategy == 'dense':
                x = self.merge_dnn(tf.concat(xs, axis=-1))
            else:
                raise NotImplementedError
        else:
            x = xs[0]
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
