import pickle
import warnings

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding
from tensorflow import keras
from cf.layers.mask import MaskedEmbeddingsAggregator as MEA
from cf.layers.ytb import L2Norm
from cf.utils.logger import logger
from cf.utils.tensor import *
import faiss
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
        self.mask_agg = MEA(name='aggregate_embedding')
        self.query_mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])
        self.item_mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])
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
            f.name: Input(shape=(), name=f.name)
            for f in self.feature_columns
        }
        # TODO 这里的 training：true 会不会有 bug
        model = Model(inputs, outputs=self.call(inputs, training=True))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        # 使用双塔就需要区分 query 字段和 item 字段
        query_x = {k: inputs[k] for k in self.query_cols}
        query_out = self.query_tower(query_x)
        if training:
            # 训练过程，返回 loss
            item_x = {k: inputs[k] for k in self.item_cols}
            item_out = self.item_tower(item_x)
            # 相似度计算最终使用 temperature
            batch_size = tf.shape(query_out)[0]
            sim = tf.nn.softmax((query_out @ tf.transpose(item_out)) / self.temperature, axis=-1)
            loss = (-tf.math.log(sim) @ tf.eye(batch_size)) / tf.cast(batch_size, 'float32')
            # 置空 item index
            self.item_index = None
            # 保存数据
            self.add_loss(lambda: loss)
            return loss
        else:
            # predict / eval 过程，返回结果
            self.load_item_index()
            D, top_k = self.item_index.search(query_out, 100)
            batch_size = tf.shape(query_out)[0]
            r5, r10, r30, r50, r100, rc_cnt, hr_cnt = 0, 0, 0, 0, 0, 0, 0
            for i, q in enumerate(query_x[self.topk_cmp_col]):
                s1 = {a for a in q if a != 0}
                rc_cnt += len(s1)
                hr_cnt += np.sum(np.array(top_k[i]) != -1)
                r5 += len(s1.intersection(top_k[i][:5]))
                r10 += len(s1.intersection(top_k[i][:10]))
                r30 += len(s1.intersection(top_k[i][:30]))
                r50 += len(s1.intersection(top_k[i][:50]))
                r100 += len(s1.intersection(top_k[i][:100]))
            info = f'Recall@5 {r5 / rc_cnt:.4f}, Recall@10 {r10 / rc_cnt:.4f},' \
                   f'Recall@30 {r30 / rc_cnt:.4f}, Recall@50 {r50 / rc_cnt:.4f}, Recall@100 {r100 / rc_cnt:.4f}\n'
            info += f'HR@5 {r5 / min(hr_cnt, 5 * batch_size):.4f}, HR@10 {r10 / min(hr_cnt, 10 * batch_size):.4f}, ' \
                    f'HR@30 {r30 / min(hr_cnt, 30 * batch_size):.4f}, HR@50 {r50 / min(hr_cnt, 50 * batch_size):.4f}, ' \
                    f'HR@100 {r100 / min(hr_cnt, 50 * batch_size):.4f}\n'
            logger.info(info)
            return top_k

    def query_tower(self, x):
        x = self.form_x(x)
        x = self.query_mlp(x)
        return tf.math.l2_normalize(x)

    def item_tower(self, x):
        x = self.form_x(x)
        x = self.item_mlp(x)
        return tf.math.l2_normalize(x)

    def form_x(self, x):
        # x - (batch, dim)
        sparse_x = []
        dense_x = []
        seq_x = []
        for f, v in x.items():
            key = f
            if '::' in f:  # Get the key
                key = f.split('::')[1]
            if f[0] == 'C':
                sparse_x.append(self.ebd[key](v))
            elif f[0] == 'I':
                dense_x.append(tf.expand_dims(v, 1))
            elif f[0] == 'S':
                seq_x.append(tf.expand_dims(self.mask_agg(self.ebd[key](v)), 1))
            else:
                warnings.warn(f'The feature:{f} may not be categorized', SyntaxWarning)
        # TODO Debug shape error
        return tf.concat(sparse_x + dense_x + seq_x, axis=-1)

    def save_vector(self):
        # pickle.dump(self.query_data, open(os.path.join(self.directory, 'query.pk'), 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.item_data, open(os.path.join(self.directory, 'item.pk'), 'wb'), pickle.HIGHEST_PROTOCOL)
        logger.info(f'Save item vectors to {os.path.abspath(self.directory)}')

    def load_item_index(self):
        if self.item_index is not None:
            return
        if self.item_data is None or len(self.item_data) == 0:
            self.item_data = pickle.load(open(os.path.join(self.directory, 'item.pk'), 'rb'))

        ids = self.item_data.keys()
        values = np.array([self.item_data[k] for k in ids])
        dim = 0
        if values.shape == 2:
            dim = values[1]
        else:
            e = f'The values dimension must be 2.'
            logger.error(e)
            raise ValueError(e)
        quan = faiss.IndexFlatIP(dim)
        nlist = int(values.shape[0] / 10)
        nlist = nlist if nlist > 0 else 1
        self.item_index = faiss.IndexIVFFlat(quan, dim, nlist)
        self.item_index.train(values)
        self.item_index.add_with_ids(values, ids)
