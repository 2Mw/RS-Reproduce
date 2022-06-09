import os
from cf.utils.logger import logger
import tensorflow as tf
from pynvml.smi import nvidia_smi as smi
import time
from cf.utils.config import get_date, get_random_num


def get_available_gpu(ins):
    g = ''
    res = ins.DeviceQuery('memory.free')
    for idx, i in enumerate(res['gpu']):
        if i['fb_memory_usage']['free'] > MEMORY_LIMIT:
            g = f'{idx}'
            break
    return g


MEMORY_LIMIT = 1024  # MiB
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
