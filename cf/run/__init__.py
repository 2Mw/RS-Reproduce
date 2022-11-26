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
from cf.run.run_dcn_me import *
from cf.run.run_autoint_me import *
from cf.run.run_youtubesbc import *
from cf.run.run_youtubednn_recall import *
from cf.run.run_doubletower import *
from cf.run.run_mind import *
from cf.run.run_mime import *
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
        self.predict = getattr(module, 'predict')


MODULES = {k: Instance(k) for k in cf.models.MODULES.keys()}

# Use to select GPU.

MEMORY_LIMIT_RATIO = 0.7  # MiB
USE_CPU_ONLY = False
ASSIGNED_GPU = os.environ.get("CUDA_VISIBLE_DEVICES")


def get_available_gpu(ins, stringify=False):
    """
    Return a list of available gpus,

    :param ins: The instance of smi
    :param stringify: Weather you want to stringify list.
    :return: "0", "0,1", "0,2,3" or [0], [0,1]
    """
    g = []
    res = ins.DeviceQuery('memory.free,memory.total')
    for idx, i in enumerate(res['gpu']):
        if i['fb_memory_usage']['free'] / i['fb_memory_usage']['total'] >= MEMORY_LIMIT_RATIO:
            g.append(idx)
    g = list(map(str, g))
    if stringify:
        return ','.join(g)
    else:
        return g


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    logger.info(f'You selected CPU only.')
else:
    ins = smi.getInstance()
    while True:
        SELECTED_GPU = get_available_gpu(ins)
        if len(SELECTED_GPU) == 0 or (ASSIGNED_GPU is not None and ASSIGNED_GPU not in ','.join(SELECTED_GPU)):
            print(f'\rCurrently GPUs are busy possibly({get_date()}).', end="")
            time.sleep(get_random_num(5, 8))
        else:
            # time.sleep(get_random_num(10, 15))
            a = get_available_gpu(ins)
            if ASSIGNED_GPU is None:
                if len(a) == 0 or len(SELECTED_GPU) == 0:
                    continue
                if a[0] == SELECTED_GPU[0]:  # 只选取一个GPU
                    ASSIGNED_GPU = SELECTED_GPU[0]
                    break
            else:
                if ASSIGNED_GPU in ','.join(a):
                    break
    os.environ["CUDA_VISIBLE_DEVICES"] = ASSIGNED_GPU
    # 设置 gpu 现存使用策略
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)

    logger.info(f'You selected gpu:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
