import os
import cf
from cf.utils.config import get_date
import yaml
from cf.config import MODULES as pool
from cf.utils.logger import logger
import argparse
import sys

'Use to generate various hyper parameters file for fine tune model'

_pr = os.path.join(cf.get_project_path(), 'cf')
_tune_path = os.path.abspath(os.path.join(_pr, 'tune'))
if not os.path.exists(_tune_path):
    os.mkdir(_tune_path)


def gen_tunes(model, cfg, part, item, opt_values=None):
    """
    The generate file path is cf/tune/model/{timestamp}/x.yaml
    if opt_values = [1,2,3]
    This will enumerate opt_values for cfg[part][item],
    then generate three configs for:
        cfg[part][item]=1
        cfg[part][item]=2
        cfg[part][item]=3
    if opt_value is None
    """
    path = _tune_path
    dirs = [model, get_date()]
    for i in dirs:
        path = os.path.join(path, i)
        if not os.path.exists(path):
            os.mkdir(path)

    if opt_values is None or len(opt_values) == 0:
        with open(os.path.join(path, f'0.yaml'), 'w') as f:
            yaml.dump(cfg, f)
    else:
        for idx, obj in enumerate(opt_values):
            cfg[part][item] = obj
            with open(os.path.join(path, f'{idx}.yaml'), 'w') as f:
                yaml.dump(cfg, f)
    print(f'Output all yaml files to {path}')


def cli():
    p = argparse.ArgumentParser(description="Generate yaml.")
    p.add_argument('-m', '--model', required=True, help='The model name')
    return p.parse_args()


if __name__ == '__main__':
    arg = cli()
    name = arg.model.lower()
    if pool.get(name) is None:
        logger.error(f'The model:{name} not exists.')
        sys.exit(0)
    gen_tunes(name, pool.get(name), 'train', 'batch_size')
