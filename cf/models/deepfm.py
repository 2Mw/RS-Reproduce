import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from cf.layers import fm, mlp
from keras.layers import Input


class DeepFM(Model):
    def __init__(self, feature_columns, config):
        """

        :param feature_columns:  A list. [{'name':, 'feature_num':, 'dim':}, ...]
        :param config:  Hyper parameters configurations.
        """
        super(DeepFM, self).__init__()
        self.feature_columns = feature_columns
        # Load configuration
        self.config = config
        self.model_cfg = config['model']
        self.training_cfg = config['train']
        # Load parameters
        self.embedding_dim = self.model_cfg['embedding_dim']
        self.map_dict = {}
        self.feature_len = 0
        self.field_num = len(feature_columns)
        for feature in feature_columns:
            self.map_dict[feature['name']] = feature['feature_num']
            self.feature_len += feature['feature_num']
        # Layer initialization
        self.embedding_layers = {
            feature['name']: keras.layers.Embedding(
                input_dim=feature['feature_num'],
                input_length=1,
                output_dim=feature['dim'],
                embeddings_initializer='random_normal',
                embeddings_regularizer=keras.regularizers.l2(self.model_cfg['embedding_reg']),
            )
            for feature in feature_columns
        }
        self.fm = fm.FMLayer(self.feature_len, self.model_cfg['fm_w_reg'])
        self.mlp = mlp.MLP(self.model_cfg['hidden_units'], self.model_cfg['activation'], self.model_cfg['dropout'])
        self.dense = keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, **kwargs):
        # embedding, (batch_size, embedding_dim*fields)
        group = []

        for feature_name, value in inputs.items():
            group.append(self.embedding_layers[feature_name](value))
        sparse_embedding = tf.concat(group, axis=-1)
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
        inputs = {
            feature['name']: keras.layers.Input(shape=(), dtype=tf.int32, name=feature['name'])
            for feature in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
    #
    # def build(self, input_shape):
    #     inputs = Input(input_shape)
    #     outputs = self.call(inputs)
    #     self.model_plot = tf.keras.Model(inputs=inputs, outputs=outputs)
    #     print("build")
    #     super(DeepFM, self).build()
