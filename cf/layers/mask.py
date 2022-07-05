import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Layer, LayerNormalization, Dropout
from keras.regularizers import l2
from cf.utils.logger import logger


class MaskBlock(Layer):
    def __init__(self, hidden_dim, output_dim, reduction_ratio, dropout_rate=0, **kwargs):
        """
        The mask block in masknet.

        :param hidden_dim: The dim of aggregation layer.
        :param output_dim: The dim of projection layer
        :param reduction_ratio: The ratio of projection dim and aggregation dim.
        :param dropout_rate: The dropout rate.
        :param kwargs:
        """
        super().__init__(**kwargs)
        aggregation_dim = int(hidden_dim * reduction_ratio)
        self.mask_layer = [Dense(aggregation_dim, 'relu'), Dense(hidden_dim)]
        self.hidden_layer = Dense(output_dim, use_bias=False)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        if len(args) != 1:
            e = f'The arguments is invalid'
            logger.error(e)
            raise ValueError(e)
        v_mask = self.mask_layer[1](self.mask_layer[0](inputs))
        v_out = self.dropout(tf.nn.relu(self.layer_norm(self.hidden_layer(v_mask * args[0]))))
        return v_out
