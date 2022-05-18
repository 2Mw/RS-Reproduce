import json
import os
import random
import time
import numpy as np
import yaml
import sys
import cf
from keras.api._v2 import keras

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
        print(f'Export config file to {path} successfully.')


def get_config(model: str, timestamp: str):
    """
    Get configuration from specified yaml file. The filepath is ../result/{model}/{timestamp}/config.yaml

    :param model: The model name
    :param timestamp: the filepath of yaml file.
    :return: The Dict.
    """
    file = f'../result/{model}/{timestamp}/config.yaml'
    if not os.path.exists(file):
        raise FileExistsError(f"The file {file} not exists.")
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


def export_result(train_hist: keras.callbacks.History, val_res, directory: str, cost: float, params: int):
    """
    Export model training result to specified file. {directory}/result.txt

    :param params: The count of model parameters
    :param cost: cost seconds.
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param directory: the directory to export.
    :return:
    """
    info = {
        'cost_seconds': cost,
        "params": params,
        'train': {
            'epochs': train_hist.params['epochs'],
            'history': train_hist.history
        },
        'test': {
            'result': val_res
        }
    }
    f = open(os.path.join(directory, 'result.txt'), 'w')
    json.dump(info, f)
    f.close()


def export_all(directory: str, config: object, model: keras.models.Model, train_hist: keras.callbacks.History, val_res,
               cost):
    """
    Export all information of model.

    :param directory: The directory to export.
    :param config: The hyper parameters.
    :param model: The model of class `keras.models.Model`
    :param train_hist: train_hist from model.fit()
    :param val_res: test result from model.evaluate()
    :param cost: cost seconds.
    :return:
    """
    export_config(config, directory)
    model.save_weights(f'{directory}/weights_last.h5')
    export_result(train_hist, val_res, directory, cost, model.count_params())
    # keras.utils.plot_model(model.build_graph(), os.path.join(directory, 'model.png'))
    # keras.utils.plot_model(model.model_plot, os.path.join(directory, 'model.png'))
    print(f'Successfully export all information of model to {os.path.abspath(directory)}')


if __name__ == '__main__':
    print(__name__)
    print(os.path.abspath(".."))
    print(sys.modules[__name__])
    print(get_date())
    print(__file__)
    print(cf.get_project_path())
