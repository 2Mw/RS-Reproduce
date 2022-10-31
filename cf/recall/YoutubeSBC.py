import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from cf.layers import crossnet, mlp, gate, moe
from cf.utils import tensor
from cf.models.base import get_embedding, form_x, model_summary
from cf.preprocess.feature_column import SparseFeat, SequenceFeat
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip
from tensorflow import keras


class YoutubeSBC(Model):
    def __init__(self, user_features, item_features, sample_weight_feature, config, directory,
                 *args,
                 **kwargs):
        """
        Sampling-Bias-Corrected Neural Modeling for Large Corpus.
        It's a DSSM match model trained by In-batch softmax loss on list-wise samples, and add sample debias module

        :param user_features: training by user tower <user, context>
        :param item_features: training by item tower <item>
        :param sample_weight_feature: used forr sampling bias corrected in training.
        :param user_params: the params of user tower module. {"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        :param item_params: the param of item tower module.
        :param config: the config of model
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # get config
        self.directory = directory
        train_cfg = config['train']
        model_cfg = config['model']
        self.user_features = user_features
        self.item_feature = item_features
        self.sample_weight_features = sample_weight_feature
        self.user_dims = sum([f.embed_dim for f in user_features])
        self.item_dims = sum([f.embed_dim for f in item_features])
        self.batch_size = train_cfg['batch_size']

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        pass

    def user_tower(self, x):
        pass

    def item_tower(self, x):
        pass
