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
from keras.losses import cosine_similarity


class YoutubeDNN(Model):
    def __init__(self, user_features, item_features, neg_item_features, config, directory="", *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Get config
        self.directory = directory
        model_cfg = config['model']
        train_cfg = config['train']
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_features = neg_item_features
        self.temperature = model_cfg['temperature']
        self.embedding_dim = model_cfg['embedding_dim']
        # tower params
        utp = model_cfg['user_params']  # user tower params
        itp = model_cfg['item_params']  # item tower params
        # embedding
        self.ebd = get_embedding(user_features + item_features, self.embedding_dim)
        # mlp
        self.user_mlp = mlp.MLP(utp['units'], utp['activation'], utp['dropout'], utp['use_bn'])

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)

    def call(self, inputs, training=None, mask=None):
        # 首先将 inputs 分为 user_input 和 item_input
        user_names = [f.name for f in self.user_features]
        item_names = [f.name for f in self.item_features]
        neg_names = [f.name for f in self.neg_item_features]
        user_input = {k: v for k, v in inputs.items() if k in user_names}
        item_input = {k: v for k, v in inputs.items() if k in item_names}
        neg_input = {k: v for k, v in inputs.items() if k in neg_names}
        # Get vector
        user_vec = self.user_tower(user_input)
        item_vec = self.item_tower(item_input, neg_input)
        y = cosine_similarity(user_vec, item_vec)
        return y / self.temperature

    def user_tower(self, x):
        user_x = form_x(x, self.ebd, False)
        return self.user_mlp(user_x)

    def item_tower(self, x, neg):
        item_x = form_x(x, self.ebd, False)
        neg_x = form_x(neg, self.ebd, False)
        return tf.concat([item_x, neg_x], axis=-1)
