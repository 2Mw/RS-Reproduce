import os
from cf.utils.logger import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    SELECTED_GPU = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
logger.info(f'You selected gpu:{os.environ.get("CUDA_VISIBLE_DEVICES")}')
