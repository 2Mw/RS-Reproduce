from keras.layers import Layer


class Linear(Layer):
    def __init__(self, feature_column, **kwargs):
        super().__init__(**kwargs)
        self.feature_column = feature_column
