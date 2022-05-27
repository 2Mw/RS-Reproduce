import json
import os.path

import numpy as np
from tensorflow.keras.callbacks import Callback
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


class AbnormalAUC(Callback):
    def __init__(self, threshold=0.8, steps: int = 0, directory: str = ''):
        """
        用于训练fit过程早停，auc大于threshold就停止训练

        :param threshold: train auc的最大值
        :param steps: 大于某个steps才生效
        :param directory: Use to save model for lower tf version.
        """
        super(AbnormalAUC, self).__init__()
        self._supports_tf_logs = True
        self.threshold = threshold
        self.steps = steps
        self.directory = directory
        self.low_tf_version = float(tf.__version__.replace('.', '')) < 240

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        auc = logs.get('auc')
        if auc is not None:
            auc = auc.numpy().item()
            if auc > self.threshold and batch > self.steps:
                self.model.stop_training = True
                if self.low_tf_version:
                    self.model.save_weights(os.path.join(self.directory, f'weights.999-{self.threshold}.hdf5'))


class MetricsMonitor(Callback):
    def __init__(self, metric, mode: str, directory: str = '', sample_step: int = 3):
        """
        每个 train_batch_end，观察对应基准指标的最佳值以及变化趋势

        :param metric: 想要监控的指标
        :param mode: 监控模式，max or min
        :param directory: 输出变化趋势到对应目录
        """
        super(MetricsMonitor, self).__init__()
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
            if type(value) != type(1.0):
                value = value.numpy().item()
            if not np.isnan(value) and not np.isinf(value) and self.op(value, self.best_value):
                self.best_value = value
                self.best_epoch = self.epoch
            if self.directory != '' and batch % self.sample_step == 0:
                self.records.append(value)

    def on_epoch_begin(self, epoch, logs=None):
        self.records = []
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        if self.directory != '' and os.path.exists(self.directory):
            with open(os.path.join(self.directory, f'{self.metric}_{epoch}.pickle'), 'wb') as f:
                pickle.dump(self.records, f, pickle.HIGHEST_PROTOCOL)
            plt.figure()
            plt.plot(self.records)
            # plt.legend(f'epoch-{epoch}')
            plt.title(f'epoch-{epoch}')
            if epoch == 0:
                plt.ylim(0.785, 0.810)
            plt.xlabel('steps')
            plt.ylabel(self.metric)
            plt.savefig(os.path.join(self.directory, f'{self.metric}_curve_{epoch}.png'))

    def on_train_end(self, logs=None):
        res = {f'{self.metric}_{self.mode}': self.best_value}
        if self.directory != '' and os.path.exists(self.directory):
            with open(os.path.join(self.directory, f'{self.metric}_{self.mode}.json'), 'w') as f:
                json.dump(res, f)
        print(f'{self.metric}_{self.mode} best value is {self.best_value} in epoch-{self.best_epoch}')
