import os
import pickle
import tensorflow as tf
import cf
from cf.models import MODULES as pool
from cf.preprocess.criteo import *
from cf.utils.logger import logger
from tensorflow import keras

project_dir = cf.get_project_path()


def initModel(model_name: str, cfg, feature_columns, directory, weights: str = ''):
    """
    有机的加载模型，可以从已有权重中继续训练模型

    :param model_name: The name of model
    :param cfg:
    :param feature_columns:
    :param directory:
    :param weights: 加载权重，空表示不提前加载权重
    :return:
    """
    train_config = cfg['train']
    model_config = cfg['model']
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    ins = pool.get(model_name)
    model = ins(feature_columns, cfg, directory)
    model.summary()
    optimizer = get_optimizer(train_config['optimizer'], train_config['lr'])
    model.compile(loss=train_config['loss'], optimizer=optimizer, metrics=model_config['metrics'])
    model.compile(loss=train_config['loss'], optimizer=train_config['optimizer'], metrics=model_config['metrics'])
    if weights == '' or weights is None:
        return model
    if os.path.exists(weights):
        model.built = True
        model.load_weights(weights)
        logger.info(f'========= Load pre-train weights =========')
    else:
        raise FileNotFoundError(f'{weights} weights file not exists.')
    return model


def evaluate(model_name: str, cfg, weight: str, dataset: str = 'criteo'):
    base = os.path.join(project_dir, cfg['files'][f'{dataset}_base'])
    sample_size = cfg['train']['sample_size']
    if sample_size == -1:
        data_dir = os.path.join(base, f'data_all')
    else:
        data_dir = os.path.join(base, f'data_{sample_size}')
    if os.path.exists(data_dir):
        feature_columns = pickle.load(open(f'{data_dir}/feature.pkl', 'rb'))
        test_data = pickle.load(open(f'{data_dir}/test_data.pkl', 'rb'))
    else:
        raise FileNotFoundError(f'{data_dir} not found.')
    train_config = cfg['train']
    model = initModel(model_name, cfg, feature_columns, '', weight)
    res = model.evaluate(test_data[0], test_data[1], batch_size=train_config['test_batch_size'])
    logger.info(res)


def load_data(dataset: str, base: str, sample_size: int, test_ratio: float, train_file, embedding_dim):
    """
    Load feature columns, train data, test data from files.

    :param dataset: The name of dataset
    :param base: The base filepath of dataset.
    :param sample_size: The size of train data.
    :param test_ratio: The ratio of test data.
    :param train_file: The name of train data file.
    :param embedding_dim: The dimension of embedding.
    :return: feature_columns, train_data, test_data
    """
    if sample_size == -1:
        data_dir = os.path.join(base, f'data_all')
    else:
        data_dir = os.path.join(base, f'data_{sample_size}')
    if os.path.exists(data_dir):
        logger.info(f'读取已保存数据')
        feature_columns = pickle.load(open(f'{data_dir}/feature.pkl', 'rb'))
        train_data = pickle.load(open(f'{data_dir}/train_data.pkl', 'rb'))
        test_data = pickle.load(open(f'{data_dir}/test_data.pkl', 'rb'))
    else:
        logger.info(f'数据处理中')
        feature_columns, train_data, test_data = None, None, None
        if dataset.lower() == 'criteo':
            feature_columns, train_data, test_data = create_criteo_dataset(train_file, embedding_dim, sample_size,
                                                                           test_ratio)
        os.mkdir(data_dir)
        pickle.dump(feature_columns, open(f'{data_dir}/feature.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_data, open(f'{data_dir}/train_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, open(f'{data_dir}/test_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        logger.info(f'保存数据')
    return feature_columns, train_data, test_data


def get_optimizer(name, lr, clipnorm: float = 10):
    opt = keras.optimizers.get(name)
    opt.learning_rate = lr
    opt.clipnorm = clipnorm if clipnorm is not None else 10
    return opt
