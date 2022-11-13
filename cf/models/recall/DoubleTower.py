import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding, form_x, model_summary
from tensorflow import keras
from cf.layers.mask import MaskedEmbeddingsAggregator as MEA
from cf.layers.ytb import L2Norm
import os


class DoubleTower(Model):
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_columns = feature_columns
        model_cfg = config['model']
        self.directory = directory
        self.embedding_dim = model_cfg['embedding_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim, mask_zero=True)
        self.temperature = model_cfg['temperature']
        self.mask_agg = MEA(name='aggregate_embedding')
        self.mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])
        self.l2 = L2Norm()
        self.bn = BatchNormalization()

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            f.name: Input(shape=(None,), name=f.name)
            for f in self.feature_columns
        }
        model = Model(inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        # TODO 使用双塔就需要区分 query 字段和 item 字段
        # TODO 使用 temperature
        pass
