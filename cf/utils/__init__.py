import os

SELECTED_GPU = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
print(f'You selected gpu:{SELECTED_GPU}')