import argparse
import logging
import os
import sys

import yaml

import cf
import cf.run
from cf.models import MODULES as pool
from cf.preprocess import datasets
from cf.run import MODULES as run_pool
from cf.utils.config import get_date
from cf.utils.logger import logger

_log_file = ''

def cli():
    p = argparse.ArgumentParser(description="Train model with parser.")
    p.add_argument('-m', '--model', required=True, help='The model name')
    p.add_argument('-c', '--config', required=True, help='The path of config file.')
    p.add_argument('-d', '--dataset', type=str, default='criteo', help='The dataset\'s name.')
    p.add_argument('-p', '--pretrain', type=str, default='', help='The filepath of the weight of pretrain model.')
    p.add_argument('-t', '--type', type=str, default='train', help='Select the run')
    p.add_argument('-l', '--log', type=bool, default=True,
                   help='Whether output log info to file, default false(use stderr)') 
    return p.parse_args()


def checkArgs(args):
    """
    用于检查参数是否合法

    :param args:
    :return: 返回yaml配置对象
    """
    global _log_file
    model_name = args.model.lower()
    if pool.get(model_name) is None:
        logger.error(f'The model:{model_name} not exists.')
        sys.exit(0)

    if args.dataset.lower() not in datasets:
        logger.error(f'The dateset:{args.dataset} is not in list, only support these: {datasets}.')
        sys.exit(0)

    if args.log:  # 是否输出日志到文件
        _log_file = os.path.join(cf.get_project_path(), 'log')
        if not os.path.exists(_log_file):
            os.mkdir(_log_file)

        _log_file = os.path.join(_log_file, f'{model_name}-{args.dataset.lower()}-{get_date()}.log')
        print(f'Output log file in {_log_file}')
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', datefmt='%y%m%d%H%M%S', filename=_log_file)

    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        logger.error(f'The filepath of config:{args.config} not exists.')
        sys.exit(0)

    # config 文件必须是 yaml 文件
    suffix = cfg_path.split('.')
    if len(suffix) > 1 and (suffix[-1] == 'yaml' or suffix[-1] == 'yml'):
        # 加载 yaml 文件
        with open(cfg_path, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)
    else:
        logger.error(f'The config file must be the suffix of `.yaml` or `yml`')
        sys.exit(0)

    if len(args.pretrain) > 0:
        weight_path = os.path.abspath(args.pretrain)
        if not os.path.exists(weight_path):
            logger.error(f'The filepath of weights:{args.pretrain} not exists.')
            sys.exit(0)

    mode = args.type.lower()
    if mode not in ['train', 'test']:
        logger.error(f'The mode: {args.type} is invalid.')
        sys.exit(0)

    if mode == 'test' and args.pretrain == '':
        # 测试必须指定预训练权重
        logger.error(f'The `pretrain` weight is required if mode is test.')
        sys.exit(0)

    return config


if __name__ == '__main__':
    # 1. 构建参数信息
    arg = cli()
    # 2. 检查参数
    cfg = checkArgs(arg)
    # 3. 解析参数
    name = arg.model.lower()
    model = run_pool.get(name)
    if model is None:
        e = f'The module: {name} is not found'
        logger.error(e)
        raise ModuleNotFoundError(e)

    dataset = arg.dataset.lower()
    if arg.type.lower() == 'train':
        model.train(cfg, dataset, arg.pretrain)
    else:
        model.evaluate(cfg, arg.pretrain, dataset)
    print(f'Output log file in {_log_file}')

# print(arg.config)
