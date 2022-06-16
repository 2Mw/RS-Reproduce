import cf.models
import cf.run as obj
from cf.run.run_autoint import *
from cf.run.run_can import *
from cf.run.run_dcn import *
from cf.run.run_dcnv2 import *
from cf.run.run_deepfm import *
from cf.run.run_interhat import *
from cf.run.run_edcn import *
from cf.run.run_medcn import *
import os
from cf.utils.logger import logger
import tensorflow as tf
from pynvml.smi import nvidia_smi as smi
import time
from cf.utils.config import get_date, get_random_num


class Instance:
    def __init__(self, name):
        module = getattr(obj, f'run_{name}')
        self.train = getattr(module, 'train')
        self.evaluate = getattr(module, 'evaluate')


MODULES = {k: Instance(k) for k in cf.models.MODULES.keys()}

# Use to select GPU.

MEMORY_LIMIT = 8192  # MiB
USE_CPU_ONLY = False


def get_available_gpu(ins, num=1):
    """
    Return the available numbers of gpus,

    :param ins: The instance of smi
    :param num: The number of gpu you want to select.
    :return: "0", "0,1", "0,2,3"
    """
    g = ''
    res = ins.DeviceQuery('memory.free')
    for idx, i in enumerate(res['gpu']):
        if i['fb_memory_usage']['free'] > MEMORY_LIMIT:
            g += f'{idx},'
            if len(g) / 2 == num:
                break
    return g[0:-1]


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    logger.info(f'You selected CPU only.')
else:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        SELECTED_GPU = ''
        ins = smi.getInstance()
        while SELECTED_GPU == '':
            SELECTED_GPU = get_available_gpu(ins)
            if SELECTED_GPU == '':
                print(f'\rCurrently GPUs are busy possibly({get_date()}).', end="")
                time.sleep(get_random_num(5, 8))
            else:
                time.sleep(get_random_num(3, 5))
                a = get_available_gpu(ins)
                if a == SELECTED_GPU:
                    break
                else:
                    SELECTED_GPU = ''
        os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
    # 设置 gpu 现存使用策略
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logger.info(f'You selected gpu:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
