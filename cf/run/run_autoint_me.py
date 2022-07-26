import copy
import os.path
from cf.config.autoint_me import config
from cf.utils.config import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from cf.utils.callbacks import AbnormalAUC, MetricsMonitor
import cf.run.base as base
from cf.preprocess import data as dataloader
from cf.utils.logger import logger
import cf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = cf.get_project_path()

__model__ = 'autoint_me'


def train(cfg, dataset: str = 'criteo', weights: str = ''):
    bcfg = copy.deepcopy(cfg)
    start = time.time()
    logger.info(f'========= Loading configures of {__model__} =========')
    basepath = os.path.join(project_dir, cfg['files'][f'{dataset}_base'])
    train_file = os.path.join(basepath, cfg['files'][f'{dataset}_train'])
    train_config = cfg['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    sample_size = train_config['sample_size']
    logger.info(f'========= Loading {dataset} Data =========')
    num_process = cfg['model']['numeric_process']
    feature_columns, train_data, test_data = dataloader.load_data(dataset, basepath, sample_size,
                                                                  train_config['test_ratio'], train_file, num_process=num_process)
    logger.info(f'========= Build Model =========')
    steps = int(len(train_data[1]) / batch_size)
    # 创建输出结果目录
    directory = base.create_result_dir(__model__, project_dir)
    export_config(copy.deepcopy(bcfg), directory)
    model = initModel(cfg, feature_columns, directory, weights, steps=steps)
    # 创建回调
    ckpt = ModelCheckpoint(os.path.join(directory, 'weights.{epoch:03d}-{val_loss:.5f}.hdf5'), save_weights_only=True)
    earlyStop = EarlyStopping('val_BCE', min_delta=0.0001, patience=2)
    aucStop = AbnormalAUC(0.84, steps=steps // 2, directory=directory, gap_steps=steps // 10)
    aucMonitor = MetricsMonitor('auc', 'max', directory)
    # tb = TensorBoard(log_dir=os.path.join(directory, 'profile'), histogram_freq=10, profile_batch=[3, steps])
    train_history = model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size,
                              validation_data=test_data, callbacks=[ckpt, earlyStop, aucStop, aucMonitor])
    logger.info(f'Train result: \n{train_history.history}\n')
    res = model.evaluate(test_data[0], test_data[1], batch_size=train_config['test_batch_size'])
    res = dict(zip(model.metrics_names, res))
    logger.info(f'Result: {res}')
    logger.info('========= Export Model Information =========')
    cost = time.time() - start
    export_all(directory, bcfg, model, train_history, res, cost, dataset, weights)
    logger.info(f'========= Train over, cost: {cost:.3f}s =========')


def initModel(cfg, feature_columns, directory, weights: str = '', **kwargs):
    return base.initModel(__model__, cfg, feature_columns, directory, weights, **kwargs)


def evaluate(cfg, weight: str, dataset: str = 'criteo'):
    base.evaluate(__model__, cfg, weight, dataset)


if __name__ == '__main__':
    train(config)
    # evaluate(config, r'E:\Notes\DeepLearning\practice\rs\cf\result\autoint_me\20220617214840\weights.008-0.51019.hdf5')
