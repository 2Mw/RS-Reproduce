import gc
import json
import os.path

import numpy as np
from tensorflow.keras.callbacks import Callback
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from cf.utils.logger import logger


class AbnormalAUC(Callback):
    def __init__(self, threshold=0.8, steps: int = 0, directory: str = '', gap_steps: int = 200):
        """
        用于训练fit过程早停，auc大于threshold就停止训练

        :param threshold: train auc的最大值
        :param steps: 大于某个steps才生效
        :param directory: Use to save model for lower tf version.
        :param gap_steps: 如果保存模型，每个多少的step保存一次
        """
        super(AbnormalAUC, self).__init__()
        self._supports_tf_logs = True
        self.threshold = threshold
        self.steps = steps
        self.directory = directory
        self.low_tf_version = float(tf.__version__.replace('.', '')) < 240
        self.last_save = 1
        self.gap_steps = gap_steps

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        auc = logs.get('auc')
        if auc is not None:
            if not isinstance(auc, float) or not isinstance(auc, np.float):
                auc = auc.numpy().item()
            if auc > self.threshold and batch > self.steps:
                self.model.stop_training = True
                if self.low_tf_version:
                    if self.last_save % self.gap_steps == 0:
                        logger.warning(f"Warning: auc has exceed threshold: {self.threshold} in step {batch}.")
                        if auc < 0.82:
                            path = os.path.join(self.directory, f'weights.{self.threshold}-{auc:.5f}-{batch}.hdf5')
                            self.model.save_weights(path)
                    self.last_save += 1


class MetricsMonitor(Callback):
    def __init__(self, metric, mode: str, directory: str = '', sample_step: int = 3, dump_file: bool = False):
        """
        每个 train_batch_end，观察对应基准指标的最佳值以及变化趋势

        :param metric: 想要监控的指标
        :param mode: 监控模式，max or min
        :param directory: 输出变化趋势到对应目录
        :param sample_step: 每隔多少个batch记录一次
        :param dump_file: 是否将记录写入文件
        """
        super(MetricsMonitor, self).__init__()
        self.dump_file = dump_file
        self.directory = directory
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError(f'invalid mode: {mode}, only `min` or `max`.')
        self.mode = mode
        self.op = np.less if mode == 'min' else np.greater
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.metric = metric
        self.sample_step = sample_step
        self.records = []
        self.epoch = -1
        self.best_epoch = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        value = logs.get(self.metric)
        if value is not None:
            if not isinstance(value, float) or not isinstance(value, np.float):
                value = value.numpy().item()
            if not np.isnan(value) and not np.isinf(value) and self.op(value, self.best_value):
                self.best_value = value
                self.best_epoch = self.epoch
            if self.directory != '' and batch % self.sample_step == 0:
                self.records.append(value)

    def on_epoch_begin(self, epoch, logs=None):
        self.records = []
        gc.collect()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        if self.directory != '' and os.path.exists(self.directory):
            if self.dump_file:
                with open(os.path.join(self.directory, f'{self.metric}_{epoch}.pickle'), 'wb') as f:
                    pickle.dump(self.records, f, pickle.HIGHEST_PROTOCOL)
            plt.figure()
            l = len(self.records)
            plt.plot(np.linspace(0, l * self.sample_step, l), self.records)
            # plt.legend(f'epoch-{epoch}')
            plt.title(f'epoch-{epoch}')
            if epoch == 0:
                m = np.max(self.records)
                n = np.min(self.records)
                plt.ylim(max(n, m - 0.03), m)
            plt.xlabel('steps')
            plt.ylabel(self.metric)
            plt.savefig(os.path.join(self.directory, f'{self.metric}_curve_{epoch}.png'))

    def on_train_end(self, logs=None):
        res = {f'{self.metric}_{self.mode}': self.best_value}
        if self.directory != '' and os.path.exists(self.directory):
            with open(os.path.join(self.directory, f'{self.metric}_{self.mode}.json'), 'w') as f:
                json.dump(res, f)
        logger.info(f'{self.metric}_{self.mode} best value is {self.best_value} in epoch-{self.best_epoch}')
