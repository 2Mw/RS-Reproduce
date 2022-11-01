from keras.models import Model
from cf.models.base import checkCowclip
from cf.preprocess.feature_column import SparseFeat


class MaskNet(Model):
    def __init__(self, feature_column, config, directory="", *args, **kwargs):
        super().__init__(*args, **kwargs)
