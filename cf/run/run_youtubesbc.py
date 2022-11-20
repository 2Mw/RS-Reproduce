import copy
import os.path

import pandas as pd

from cf.config.youtubesbc import config
from cf.utils.config import *
import cf
from keras.callbacks import ModelCheckpoint
import cf.run.base as base
from cf.preprocess import data as dataloader
from cf.utils.logger import logger
import pickle
import cf.metric as metric

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = cf.get_project_path()

__model__ = 'youtubesbc'


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
    logger.info(f'========= Loading {dataset} Data =========')
    num_process = cfg['model']['numeric_process']
    feature_columns, train_data, test_user_data = dataloader.load_data(dataset, basepath, sample_size,
                                                                       train_config['test_ratio'], train_file,
                                                                       num_process=num_process, prefix='recall')

    data_dir = os.path.join(basepath, 'recall_data_all' if sample_size == -1 else f'recall_data_{sample_size}')
    item_data = pickle.load(open(f'{data_dir}/item_data.pkl', 'rb'))

    col_name = cfg['files'][f'{dataset}_columns']
    topk_cmp_col = col_name['target_id']
    drop_target = col_name['drop_target']
    if drop_target:
        train_data.pop(topk_cmp_col)
        item_data.pop(topk_cmp_col)
        test_cmp = test_user_data.pop(topk_cmp_col)
    else:
        test_cmp = test_user_data[topk_cmp_col]

    # 召回模型的多值数据预处理
    train_size = base.pad_sequence_data(train_data)
    test_size = base.pad_sequence_data(test_user_data)
    item_size = base.pad_sequence_data(item_data)

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
    train_history = model.fit(train_data, epochs=epochs, batch_size=batch_size, callbacks=[ckpt])
    # 保存模型
    model.save_weights(os.path.join(directory, 'weights.hdf5'))
    query, _ = model.predict(test_user_data, test_size)
    _, item = model.predict(item_data, item_size)
    # 得到数据，将 item 向量存入 faiss 数据库
    query_col_name = col_name['query_id']
    item_col_name = col_name['item_id']
    index = base.save_faiss(item_data[item_col_name], item, directory)
    # 确定索引开始 search
    D, top_k = index.search(query, 100)
    recalls = metric.Recall(top_k, test_cmp, 100)
    hr = metric.HitRate(top_k, test_cmp, 100)
    info = {'Recall': recalls, 'HitRate': hr}
    logger.info(info)
    res = {'dataset': dataset, 'record': info}
    logger.info('========= Export Model Information =========')
    cost = time.time() - start
    export_all(directory, bcfg, model, train_history, res, cost, dataset, weights)
    export_recall_result(test_cmp, top_k, directory)
    logger.info(f'========= Train over, cost: {cost:.3f}s =========')


def initModel(cfg, feature_columns, directory, weights: str = '', **kwargs):
    return base.initModel(__model__, cfg, feature_columns, directory, weights, **kwargs)


def evaluate(cfg, weight: str, dataset: str = 'ml100k'):
    base.recall_evaluate(__model__, cfg, weight, dataset)


def predict(cfg, weight: str, dataset: str = 'ml100k'):
    base.predict(__model__, cfg, weight, dataset)


if __name__ == '__main__':
    # train(config, 'fliggy')
    evaluate(config, r'E:\Notes\DeepLearning\practice\rs\cf\result\doubletower\20221119133246\weights.hdf5', 'fliggy')
