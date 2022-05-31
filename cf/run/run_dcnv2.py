import copy
import os.path
from cf.config.dcnv2 import config
from cf.models.dcnv2 import *
from cf.utils.config import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from cf.utils.callbacks import AbnormalAUC, MetricsMonitor
import cf.run.base as base
from cf.utils.logger import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = cf.get_project_path()

__model__ = 'dcnv2'


def train(cfg, dataset: str = 'criteo', weights: str = ''):
    bcfg = copy.deepcopy(cfg)
    start = time.time()
    logger.info(f'========= Loading configures of {__model__} =========')
    basepath = os.path.join(project_dir, cfg['files'][f'{dataset}_base'])
    train_file = os.path.join(basepath, cfg['files'][f'{dataset}_train'])
    sample_size = cfg['train']['sample_size']
    embedding_dim = cfg['model']['embedding_dim']
    logger.info(f'========= Loading {dataset} Data =========')
    feature_columns, train_data, test_data = base.load_data(dataset, basepath, sample_size, cfg['train']['test_ratio'],
                                                            train_file, embedding_dim)
    logger.info(f'========= Build Model =========')
    # 创建输出结果目录
    date = get_date()
    dirs = [__model__, date]
    directory = os.path.join(project_dir, 'cf/result')
    for d in dirs:
        directory = os.path.join(directory, d)
        if not os.path.exists(directory):
            os.mkdir(directory)
    export_config(copy.deepcopy(bcfg), directory)
    model = initModel(cfg, feature_columns, directory, weights)
    # 创建回调
    ckpt = ModelCheckpoint(os.path.join(directory, 'weights.{epoch:03d}-{val_loss:.5f}.hdf5'), save_weights_only=True)
    earlyStop = EarlyStopping(min_delta=0.0001, patience=1)
    aucStop = AbnormalAUC(0.82, steps=2000, directory=directory, gap_steps=800)
    aucMonitor = MetricsMonitor('auc', 'max', directory)

    train_config = cfg['train']

    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    steps = int(sample_size * 0.86 / batch_size)
    tb = TensorBoard(log_dir=os.path.join(directory, 'profile'), histogram_freq=10, profile_batch=[1, steps])
    train_history = model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size,
                              validation_split=train_config['val_ratio'],
                              callbacks=[ckpt, earlyStop, aucStop, aucMonitor, tb])
    res = model.evaluate(test_data[0], test_data[1], batch_size=train_config['test_batch_size'])
    logger.info(f'test AUC: {res[1]}')
    logger.info('========= Export Model Information =========')
    cost = time.time() - start
    export_all(directory, bcfg, model, train_history, res, cost)
    logger.info(f'========= Train over, cost: {cost:.3f}s =========')


def initModel(cfg, feature_columns, directory, weights: str = ''):
    return base.initModel(__model__, cfg, feature_columns, directory, weights)


def evaluate(cfg, weight: str, dataset: str = 'criteo'):
    base.evaluate(__model__, cfg, weight, dataset)


if __name__ == '__main__':
    train(config)
    # evaluate(config, r'E:\Notes\DeepLearning\practice\rs\cf\result\can\20220524195603\weights.001-0.46001.hdf5')
