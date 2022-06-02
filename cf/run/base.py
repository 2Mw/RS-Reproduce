import os
import pickle

import keras.optimizers
import tensorflow as tf
import cf
from cf.models import MODULES as pool
import tensorflow_addons as tfa
from cf.utils.logger import logger
from cf.utils.config import get_date
from cf.models.cowclip import Cowclip

project_dir = cf.get_project_path()


def initModel(model_name: str, cfg, feature_columns, directory, weights: str = '', **kwargs):
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
    cowclip = train_config.get('cowclip')
    opt = train_config['optimizer']
    lr = train_config['lr']
    lr_embed, warmup = 0, 0
    if cowclip is True:
        lr_embed = train_config['lr_embed']
        warmup = train_config['warmup']

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        ins = pool.get(model_name)
        model = ins(feature_columns, cfg, directory)
        model.summary()
        if cowclip:
            if not isinstance(model, Cowclip):
                e = 'The model is not subclass of Cowclip but set cowclip true.'
                logger.error(e)
                raise RuntimeError(e)
            # warmup 的步数
            steps = kwargs.get('steps')
            if steps is None:
                e = '`steps` is not set when cowclip is True'
                logger.error(e)
                raise ValueError(e)
            optimizers = get_optimizer(opt, lr, lr_embed, int(steps), warmup, cowclip)
            # map layer and optimizers
            layers = [
                # TODO 对于layer的名称需要进行修改
                [x for x in model.layers if "sparse_emb_" in x.name or "linear0sparse_emb_" in x.name],
                [x for x in model.layers if "sparse_emb_" not in x.name and "linear0sparse_emb_" not in x.name],
            ]
            optimizers_and_layers = list(zip(optimizers, layers))
            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        else:
            optimizer = get_optimizer(opt, lr)
        loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        model.compile(loss=loss, optimizer=optimizer, metrics=model_config['metrics'])
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


def create_result_dir(name, project_dir):
    date = get_date()
    dirs = [name, date]
    directory = os.path.join(project_dir, 'cf/result')
    for d in dirs:
        directory = os.path.join(directory, d)
        if not os.path.exists(directory):
            os.mkdir(directory)
    return directory


def get_optimizer(name, lr, lr_emb=None, steps=None, warmup=False, cowclip=False, clipnorm: float = 10):
    """
    Get optimizer for name. If cowclip is True, lr_emb is required.

    :param name: optimizer name
    :param lr: learning rate
    :param lr_emb: learning rate for embedding
    :param steps: steps per epoch if cowclip is true
    :param cowclip: if True, use cowclip to train, return a list of optimizers for embedding and dense training.
    :param clipnorm: clipnorm
    :return:
    """
    if cowclip:
        if lr_emb is None or steps is None:
            e = f'You must set lr_embed and steps if cowclip is True'
            logger.error(e)
            raise ValueError(e)
        if name.lower() != 'adam':
            e = f'Cowclip only support optmizer adam, so we change optimizer to adam.'
            logger.warning(e)
        lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-8, steps, lr, power=1) if warmup else lr
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr_emb),
            tf.keras.optimizers.Adam(learning_rate=lr_fn)
        ]
        return optimizers
    else:
        opt = keras.optimizers.get(name)
        opt.learning_rate = lr
        # opt.clipnorm = clipnorm if clipnorm is not None else 10
        return opt
