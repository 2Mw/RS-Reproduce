import json
import os
import time
import yaml
from tensorflow import keras
from cf.utils.logger import logger
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'The utils of configuration export and import'


def export_config(obj, directory: str):
    """
    export model's configuration to {directory}/config.yaml

    :param obj: The configuration dict.
    :param directory: the export directory
    :return:
    """
    _check_path()
    path = os.path.join(directory, 'config.yaml')
    with open(path, "w") as f:
        yaml.dump(obj, f)
        # print(f'Export config file to {path} successfully.')


def get_config(model: str, timestamp: str):
    """
    Get configuration from specified yaml file. The filepath is ../result/{model}/{timestamp}/config.yaml

    :param model: The model name
    :param timestamp: the filepath of yaml file.
    :return: The Dict.
    """
    file = f'../result/{model}/{timestamp}/config.yaml'
    if not os.path.exists(file):
        e = f"The file {file} not exists."
        logger.error(e)
        raise FileExistsError(e)
    with open(f'{file}', "r") as f:
        return yaml.load(f, yaml.SafeLoader)


def _check_path():
    if not os.path.exists(f'../config'):
        os.mkdir(f'../config')


def get_date() -> str:
    """
    Get the current date.

    :return: example - 20220517110856
    """
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_random_num(l, r) -> float:
    """
    Get a random float from [l, r]
    """
    if r <= l:
        raise ValueError(f'r must greater than l')
    return random.random() * (r - l) + l


def export_result(train_hist, val_res, directory: str, cost: float, model, dataset, weight, pred=None, **kwargs):
    """
    Export model training result to specified file. {directory}/result.json

    :param model: The model of class `keras.models.Model`
    :param cost: cost seconds.
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param directory: the directory to export.
    :param dataset: the dataset name.
    :param weight: the pretrain model's weight.
    :param pred: the prediction of models
    :return:
    """
    info = {
        'dataset': dataset,
        'cost_seconds': cost,
        'weight': weight,
        "params": {
            'builtin': model.count_params(),
            'mine': num_params(model),
        },
        'now': get_date(),
        'train': {
            'epochs': train_hist.params['epochs'],
            'history': train_hist.history
        },
        'test': {
            'result': val_res
        },
        **kwargs
    }

    plot_curve(train_hist.history, directory)
    f = open(os.path.join(directory, 'result.json'), 'w')
    json.dump(info, f)
    f.close()
    if pred is not None:
        pred = pd.DataFrame(pred)
        filename = os.path.join(directory, 'pred.csv')
        if dataset == 'huawei':
            pred = pred.sort_values('log_id')
            pred.to_csv(filename, index=False, header=['log_id', 'pctr'])
        else:
            pred.to_csv(filename, index=False)


def plot_curve(history, directory):
    if history.get('BCE') is None or history.get('auc') is None:
        return
    bce_df = pd.DataFrame(
        {
            'BCE': history['BCE'],
            'val_BCE': history['val_BCE'],
        }
    )

    loss_df = pd.DataFrame(
        {
            'AUC': history['auc'],
            'val_auc': history['val_auc'],
        }
    )
    plt.figure()
    sns.lineplot(data=bce_df).get_figure().savefig(os.path.join(directory, 'BCE_curve'), dpi=1000)
    plt.figure()
    sns.lineplot(data=loss_df).get_figure().savefig(os.path.join(directory, 'AUC_curve'), dpi=1000)


def export_all(directory: str, config: object, model: keras.models.Model, train_hist: keras.callbacks.History, val_res,
               cost, dataset, weight, pred=None, **kwargs):
    """
    Export all information of model.

    :param directory: The directory to export.
    :param config: The hyper parameters.
    :param model: The model of class `keras.models.Model`
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param cost: cost seconds.
    :param dataset: the dataset name.
    :param weight: the pretrain model's weight.
    :param pred: The prediction of models
    :return:
    """
    export_config(config, directory)
    export_result(train_hist, val_res, directory, cost, model, dataset, weight, pred, **kwargs)
    logger.info(f'Successfully export all information of model to {os.path.abspath(directory)}')


def export_recall_result(origin, top_k, directory):
    """
    导出召回模型的预测结果

    :param origin: 原始需要预测用户的数据
    :param top_k: 模型最终的预测结果
    :param directory: The directory to export.
    :return:
    """
    origin = pd.DataFrame(origin).values.tolist()
    top_k = pd.DataFrame(top_k).values.tolist()
    for i, e in enumerate(origin):
        origin[i] = sorted(e)
    for i, e in enumerate(top_k):
        top_k[i] = sorted(e)
    origin = pd.DataFrame(origin)
    top_k = pd.DataFrame(top_k)
    directory = os.path.join(directory, 'cmp')
    if not os.path.exists(directory):
        os.mkdir(directory)
    origin.to_csv(os.path.join(directory, 'origin.csv'), index=False)
    top_k.to_csv(os.path.join(directory, 'top_k.csv'), index=False)


def num_params(model: keras.models.Model):
    total, embed, dense = 0, 0, 0
    for v in model.trainable_variables:
        shape = v.get_shape()
        cnt = 1
        for dim in shape:
            cnt *= dim
        total += cnt
        if 'embedding' in v.name:
            embed += cnt
        else:
            dense += cnt

    res = {'total': total, 'embedding': embed, 'dense': dense}
    return res
