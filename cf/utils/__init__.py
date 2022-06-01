import os
from cf.utils.logger import logger
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    SELECTED_GPU = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
# 设置 gpu 现存使用策略
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logger.info(f'You selected gpu:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
