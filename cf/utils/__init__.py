import os
from cf.utils.logger import logger
import tensorflow as tf
from pynvml.smi import nvidia_smi as smi

MEMORY_LIMIT = 8192  # MiB
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    SELECTED_GPU = ''
    if SELECTED_GPU == '':
        ins = smi.getInstance()
        res = ins.DeviceQuery('memory.free')
        for idx, i in enumerate(res['gpu']):
            if i['fb_memory_usage']['free'] > MEMORY_LIMIT:
                SELECTED_GPU = f'{idx}'
                break
        if SELECTED_GPU == '':
            logger.warning('Currently GPUs are busy possibly.')
            SELECTED_GPU = '0'

    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
# 设置 gpu 现存使用策略
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logger.info(f'You selected gpu:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
