import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from cf.layers import fm, mlp
from cf.models.base import get_embedding, model_summary, form_x
from cf.models.cowclip import Cowclip
from cf.models.base import checkCowclip


class DeepFM(Cowclip):
    def __init__(self, feature_columns, config, directory: str = '', *args, **kwargs):
        # TODO 需要修复
        """

        :param feature_columns:  A list. [{'name':, 'vocab_size':, 'dim':}, ...]
        :param config:  Hyper parameters configurations.
        :param directory: The directory of the model.
        """
        # model params
        self.directory = directory
        self.feature_column = feature_columns
        self.config = config
        model_cfg = config['model']
        train_cfg = config['train']
        self.embedding_dim = model_cfg['embedding_dim']
        self.map_dict = {}
        self.feature_len = 0
        self.field_num = len(feature_columns)
        for f in feature_columns:
            self.map_dict[f.name] = f.vocab_size
            self.feature_len += f.vocab_size
        self.numeric_same_dim = model_cfg['numeric_same_dim']
        # cowclip params
        if train_cfg['cowclip']:
            checkCowclip(self, train_cfg['cowclip'])
            clip = train_cfg['clip']
            bound = train_cfg['bound']
            super(DeepFM, self).__init__(self.embedding_dim, clip, bound, *args, **kwargs)
        else:
            super(DeepFM, self).__init__(*args, **kwargs)
        # Layer initialization
        self.ebd = get_embedding(feature_columns, self.embedding_dim, model_cfg['embedding_device'])
        self.fm = fm.FMLayer(self.feature_len, model_cfg['fm_w_reg'])
        self.mlp = mlp.MLP(model_cfg['hidden_units'], model_cfg['activation'], model_cfg['dropout'])
        self.dense = keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, **kwargs):
        # embedding, (batch_size, embedding_dim*fields)
        sparse_embedding = form_x(inputs, self.ebd, False)
        # wide
        sparse_inputs = self._index_mapping(inputs, self.map_dict)
        wide_inputs = {
            'sparse_inputs': sparse_inputs,
            'embedding_inputs': tf.reshape(sparse_embedding, [-1, self.field_num, self.embedding_dim])
        }
        wide_outputs = self.fm(wide_inputs)
        # deep
        deep_outputs = self.mlp(sparse_embedding)
        deep_outputs = self.dense(deep_outputs)
        outputs = keras.activations.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def _index_mapping(self, inputs_dict, map_dict):
        """
        Feature index mapping

        Args:
            :param inputs_dict: A dict such as {'I1': [], 'I2': [], ...}
            :param map_dict: A dict such as {'I1': 0, 'I2': 100, ...}
        :return: new inputs tensor.
        """
        outputs_dict = {}
        for key, value in inputs_dict.items():
            if map_dict.get(key) is None:
                raise ValueError("map dict error!")
            outputs_dict[key] = tf.reshape(value + tf.convert_to_tensor(map_dict[key]), [-1, 1])
        return outputs_dict

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        model_summary(self, self.feature_column, self.directory)
    #
    # def build(self, input_shape):
    #     inputs = Input(input_shape)
    #     outputs = self.call(inputs)
    #     self.model_plot = tf.keras.Model(inputs=inputs, outputs=outputs)
    #     print("build")
    #     super(DeepFM, self).build()
