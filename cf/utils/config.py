import json
import os
import time
import yaml
from tensorflow import keras
from cf.utils.logger import logger

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


def export_result(train_hist, val_res, directory: str, cost: float, model, dataset):
    """
    Export model training result to specified file. {directory}/result.json

    :param model: The model of class `keras.models.Model`
    :param cost: cost seconds.
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param directory: the directory to export.
    :param dataset: the dataset name.
    :return:
    """
    info = {
        'dataset': dataset,
        'cost_seconds': cost,
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
        }
    }
    f = open(os.path.join(directory, 'result.json'), 'w')
    json.dump(info, f)
    f.close()


def export_all(directory: str, config: object, model: keras.models.Model, train_hist: keras.callbacks.History, val_res,
               cost, dataset):
    """
    Export all information of model.

    :param directory: The directory to export.
    :param config: The hyper parameters.
    :param model: The model of class `keras.models.Model`
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param cost: cost seconds.
    :param dataset: the dataset name.
    :return:
    """
    export_config(config, directory)
    export_result(train_hist, val_res, directory, cost, model, dataset)
    logger.info(f'Successfully export all information of model to {os.path.abspath(directory)}')


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
