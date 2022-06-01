from keras.layers import Embedding, Input, Dense
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
import os


def get_embedding(feature_columns, dim, numeric_same: bool = True, device: str = 'gpu'):
    """
    Get the embedding according to dimensions. 由于 embedding 占用参数过多，因此提供在 cpu 中训练的方法

    :param feature_columns: list of feature columns
    :param dim: Embedding dimension
    :param numeric_same: Whether to use the same embedding as categorical for numeric features
    :param device: gpu or cpu
    :return: Embedding list.
    """
    with tf.device(device.lower()):
        ebd = {}
        for f in feature_columns:
            if f['name'].startswith('C'):
                ebd[f['name']] = Embedding(input_dim=f['vocab_size'], input_length=1, output_dim=dim,
                                           embeddings_initializer='random_normal', embeddings_regularizer=l2(1e-5))
            else:
                if numeric_same:
                    # 对于数值型数据采用 autoint 中的做法
                    # ebd[f['name']] = intance.add_weight(f'{f["name"]}_embedding', shape=(1, dim), dtype=tf.float32)
                    # TODO 这么做存在问题
                    ebd[f['name']] = Dense(dim, use_bias=False)
                else:
                    # 采用 dcnv2的做法 直接将数值数据作为参数
                    ebd[f['name']] = None

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
                # mid = tf.expand_dims(v, 1) @ embedding[f]
                mid = embedding[f](tf.expand_dims(v, 1))
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
