import tensorflow as tf
from tensorflow import keras
from cf.models.cowclip import Cowclip
from keras.models import Model
from keras.layers import Dense
from cf.models.base import get_embedding, form_x, model_summary
from cf.layers import crossnet, mlp, linear, gate
from cf.utils.tensor import to2DTensor
from cf.models.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class MaskNet(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        super().__init__(*args, **kwargs)
