from keras.layers import Embedding, Input
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
import os
from cf.preprocess.feature_column import SparseFeat
from cf.models.cowclip import Cowclip
from cf.utils.logger import logger
from cf.layers.mlp import MLP


def get_embedding(feature_columns, dim, device: str = 'gpu', prefix='sparse'):
    """
    Get the embedding according to dimensions. 由于 embedding 占用参数过多，因此提供在 cpu 中训练的方法

    :param feature_columns: list of feature columns
    :param dim: Embedding dimension
    :param device: gpu or cpu
    :param prefix: The prefix of embedding name.
    :return: Embedding set, {'C1': Embedding(), 'C2': Embedding(), ... }.
    """
    with tf.device(device.lower()):
        ebd = {}
        for f in feature_columns:
            if isinstance(f, SparseFeat):
                ebd[f.name] = Embedding(input_dim=f.vocab_size, input_length=1, output_dim=dim,
                                        embeddings_initializer='random_normal', embeddings_regularizer=l2(1e-5),
                                        name=f'{prefix}_emb_{f.name}')
            else:
                ebd[f.name] = MLP([dim], None, 0, use_bn=True)

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
        f.name: Input(shape=(), dtype=tf.float32, name=f.name)
        for f in feature_column
    }
    model = Model(inputs, outputs=instance.call(inputs))
    if len(directory) > 0:
        keras.utils.plot_model(model, os.path.join(directory, 'model.png'), show_shapes=True)
    model.summary()


def form_x(inputs, embedding, divide: bool):
    """
    Generate the input `x` to the model， if attention_based is True, return (embedding_x, dense_x); if False return
    the concatenation of both.

    :param inputs:
    :param embedding: The embedding lookup set.
    :param divide: If True return value is (embedding_x, dense_x), else return the concatenation of both.
    :return:
    """
    ebd_x = []
    dense_x = []
    for f, v in inputs.items():
        if f[0] == 'C':
            ebd_x.append(embedding[f](v))
        else:
            # TODO 解决注意力机制中数值型特征 Embedding 处理
            dense_x.append(tf.expand_dims(v, 1))
    if divide:
        return tf.concat(ebd_x, axis=-1), tf.concat(dense_x, axis=-1)
    else:
        return tf.concat(ebd_x + dense_x, axis=-1)


def checkCowclip(instance, cowclip_flag):
    if isinstance(instance, Cowclip) and cowclip_flag:
        return
    else:
        e = f'The setting of cowclip mismatch. You should inherit Cowclip class and set cowclip flag `True` together.'
        logger.error(e)
        raise e


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.device('/device:cpu:0'):
        print(tf.test.gpu_device_name())
        a = tf.test.TestCase()
        a.assertEqual(tf.math.add(1, 2), 4, 'No')
