import copy
import os.path

import pandas as pd

from cf.config.youtubednn_recall import config
from cf.utils.config import *
import cf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from cf.utils.callbacks import AbnormalAUC, MetricsMonitor
import cf.run.base as base
from cf.preprocess import data as dataloader
from cf.utils.logger import logger
from keras.preprocessing.sequence import pad_sequences as ps

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = cf.get_project_path()

__model__ = 'doubletower'


def train(cfg, dataset: str = 'ml100k', weights: str = ''):
    bcfg = copy.deepcopy(cfg)
    start = time.time()
    logger.info(f'========= Loading configures of {__model__} =========')
    basepath = os.path.join(project_dir, cfg['files'][f'{dataset}_base'])
    train_file = os.path.join(basepath, cfg['files'][f'{dataset}_train'])
    train_config = cfg['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    sample_size = train_config['sample_size']
    val_ratio = train_config['val_ratio']
    logger.info(f'========= Loading {dataset} Data =========')
    num_process = cfg['model']['numeric_process']
    feature_columns, train_data, test_data = dataloader.load_data(dataset, basepath, sample_size,
                                                                  train_config['test_ratio'], train_file,
                                                                  num_process=num_process)
    train_size, test_size = 0, 0
    # 召回模型的多值数据预处理
    for k in train_data.keys():
        if k[0] == 'S':
            train_size = max(train_size, len(train_data[k]))
            train_data[k] = ps(train_data[k])

    for k in test_data.keys():
        if k[0] == 'S' and test_data[k] is not None:
            test_size = max(test_size, len(test_data[k]))
            test_data[k] = ps(test_data[k])
    # 构建模型
    logger.info(f'========= Build Model =========')
    # steps = int(len(train_data) / batch_size)
    # 创建输出结果目录
    directory = base.create_result_dir(__model__, project_dir)
    export_config(copy.deepcopy(bcfg), directory)
    cfg['dataset'] = dataset
    model = initModel(cfg, feature_columns, directory, weights)
    # 创建回调
    ckpt = ModelCheckpoint(os.path.join(directory, 'weights.{epoch:03d}.hdf5'), save_weights_only=True)
    train_history = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=test_data,
                              callbacks=[ckpt])
    model.save_vector()
    logger.info(f'Train result: \n{train_history.history}\n')
    pred = model.predict(test_data, test_size)
    res = {'dataset': dataset}
    logger.info('========= Export Model Information =========')
    cost = time.time() - start
    export_all(directory, bcfg, model, train_history, res, cost, dataset, weights, pred)
    logger.info(f'========= Train over, cost: {cost:.3f}s =========')


def initModel(cfg, feature_columns, directory, weights: str = '', **kwargs):
    return base.initModel(__model__, cfg, feature_columns, directory, weights, **kwargs)


def evaluate(cfg, weight: str, dataset: str = 'ml100k'):
    base.evaluate(__model__, cfg, weight, dataset)


def predict(cfg, weight: str, dataset: str = 'ml100k'):
    base.predict(__model__, cfg, weight, dataset)


if __name__ == '__main__':
    train(config)
    # evaluate(config, r'E:\Notes\DeepLearning\practice\rs\cf\result\can\20220524195603\weights.001-0.46001.hdf5')
