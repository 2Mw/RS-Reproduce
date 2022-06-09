#!/data/amax/b510/yl/.conda/envs/rs/bin/python
import os
import tensorflow as tf
import time
import sys
from pynvml.smi import nvidia_smi as smi

'Use to take a position of GPU'

SIZE = 50000
MEMORY_LIMIT = 8192  # MiB

if __name__ == '__main__':
    ins = smi.getInstance()
    count = ins.DeviceQuery('count')['count']
    if len(sys.argv) > 1:
        SELECTED_GPU = sys.argv[1]
    else:
        SELECTED_GPU = f'{count - 1}'

    while True:
        res = ins.DeviceQuery('memory.free')
        gpu = res['gpu'][int(SELECTED_GPU)]
        if gpu['fb_memory_usage']['free'] >= MEMORY_LIMIT:
            break
        time.sleep(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
    a = tf.ones([1, 1], dtype=tf.int32, name='123')
    fails = 0

    while True:
        s = time.time()
        try:
            a = tf.random.uniform([SIZE, SIZE], name='123')
            print('Over')
        except:
            if fails % 100 == 0:
                print('Failed')
            fails += 1
        finally:
            print(f'cost time: {time.time() - s:.2f}s')
            if a.shape[0] == SIZE:
                break
            time.sleep(0.5)

    print('Allocate Over')
    while True:
        time.sleep(100)
