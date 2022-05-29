from keras.layers import Embedding, Input, Dense
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
import os
from cf.utils.logger import logger


def get_embedding(intance: Model, feature_columns, dim, numeric_same: bool = True, device: str = 'gpu'):
    """
    Get the embedding according to dimensions. 由于 embedding 占用参数过多，因此提供在 cpu 中训练的方法

    :param intance: The model instance
    :param device: gpu or cpu
    :param feature_columns: list of feature columns
    :param dim: Embedding dimension
    :param numeric_same: Whether to use the same embedding as categorical for numeric features
    :return: Embedding list.
    """
    device = device.lower()
    device = f'/{device}:0' if device == 'cpu' else device
    print('====================')
    logger.info(f'Set embedding in device: {device}')
    with tf.device(device):
        ebd = {}
        for f in feature_columns:
            if f['name'].startswith('C'):
                ebd[f['name']] = Embedding(input_dim=f['feature_num'], input_length=1, output_dim=dim,
                                           embeddings_initializer='random_normal', embeddings_regularizer=l2(1e-5))
            else:
                if numeric_same:
                    # 对于数值型数据采用 autoint 中的做法
                    ebd[f['name']] = intance.add_weight(f'{f["name"]}_embedding', shape=(1, dim), dtype=tf.float32)
                    # ebd[f['name']] = Dense(dim, use_bias=False)
                else:
                    ebd[f['name']] = Embedding(input_dim=f['feature_num'], input_length=1, output_dim=1,
                                               embeddings_initializer='random_normal', embeddings_regularizer=l2(1e-5))

    return ebd


def model_summary(instance, feature_column, directory):
    """
    Summary the model.

    :param instance:
    :param feature_column:
    :param directory:
    :return:
    """
    inputs = {
        f['name']: Input(shape=(), dtype=tf.float32, name=f['name'])
        for f in feature_column
    }
    model = Model(inputs, outputs=instance.call(inputs))
    if len(directory) > 0:
        keras.utils.plot_model(model, os.path.join(directory, 'model.png'), show_shapes=True)
    model.summary()


def form_x(inputs, embedding, numeric_same: bool):
    x = []
    for f, v in inputs.items():
        if f[0] == 'C':
            x.append(embedding[f](v))
        else:
            if numeric_same:
                mid = tf.expand_dims(v, 1) @ embedding[f]
                # mid = embedding[f](tf.expand_dims(v, 1))
                x.append(mid)
            else:
                x.append(tf.expand_dims(v, 1))
    return tf.concat(x, axis=-1)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.device('/device:cpu:0'):
        print(tf.test.gpu_device_name())
        a = tf.test.TestCase()
        a.assertEqual(tf.math.add(1, 2), 4, 'No')
