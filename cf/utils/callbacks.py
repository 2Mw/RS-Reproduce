import json
import os.path

import numpy as np
from keras.utils import tf_utils
from keras.callbacks import Callback
import pickle
import matplotlib.pyplot as plt


class AbnormalAUC(Callback):
    def __init__(self, threshold=0.8, steps: int = 0):
        """
        用于训练fit过程早停，auc大于threshold就停止训练

        :param threshold: train auc的最大值
        :param steps: 大于某个steps才生效
        """
        super(AbnormalAUC, self).__init__()
        self._supports_tf_logs = True
        self.threshold = threshold
        self.steps = steps

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        auc = logs.get('auc')
        if auc is not None:
            auc = tf_utils.sync_to_numpy_or_python_type(auc)
            if auc > self.threshold and batch > self.steps:
                self.model.stop_training = True


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

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        value = logs.get(self.metric)
        if value is not None:
            value = tf_utils.sync_to_numpy_or_python_type(value)
            if not np.isnan(value) and not np.isinf(value) and self.op(value, self.best_value):
                self.best_value = value
            if self.directory != '' and batch % self.sample_step == 0:
                self.records.append(value)

    def on_epoch_begin(self, epoch, logs=None):
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        if self.directory != '' and os.path.exists(self.directory):
            with open(os.path.join(self.directory, f'{self.metric}_{epoch}.pickle'), 'wb') as f:
                pickle.dump(self.records, f, pickle.HIGHEST_PROTOCOL)
            plt.figure()
            plt.plot(self.records)
            # plt.legend(f'epoch-{epoch}')
            plt.title(f'epoch-{epoch}')
            plt.xlabel('steps')
            plt.ylabel(self.metric)
            plt.savefig(os.path.join(self.directory, f'{self.metric}_curve_{epoch}.png'))

    def on_train_end(self, logs=None):
        res = {f'{self.metric}_{self.mode}': self.best_value}
        if self.directory != '' and os.path.exists(self.directory):
            with open(os.path.join(self.directory, f'{self.metric}_{self.mode}.json'), 'w') as f:
                json.dump(res, f)
        print(f'{self.metric}_{self.mode} best value is {self.best_value}')
