import copy
import os.path
import pickle
from cf.config.deepfm import config
from cf.preprocess.criteo import *
import cf
import tensorflow as tf
from cf.models.deepfm import *
from cf.utils.config import *

project_dir = cf.get_project_path()

__model__ = 'deepfm'


def train(cfg, dataset: str = 'criteo'):
    bcfg = copy.deepcopy(cfg)
    start = time.time()
    print(f'========= Loading configures =========')
    base = os.path.join(project_dir, cfg['files'][f'{dataset}_base'])
    train_file = os.path.join(base, cfg['files'][f'{dataset}_train'])
    sample_size = cfg['train']['sample_size']
    embedding_dim = cfg['model']['embedding_dim']
    print(f'========= Loading {dataset} Data =========')
    data_dir = os.path.join(base, f'data_{sample_size}')
    if os.path.exists(data_dir):
        print(f'读取已保存数据')
        feature_columns = pickle.load(open(f'{data_dir}/feature.pkl', 'rb'))
        train_data = pickle.load(open(f'{data_dir}/train_data.pkl', 'rb'))
        test_data = pickle.load(open(f'{data_dir}/test_data.pkl', 'rb'))
    else:
        print(f'数据处理中')
        feature_columns, train_data, test_data = create_criteo_dataset(train_file, embedding_dim, sample_size, 0.2)
        os.mkdir(data_dir)
        pickle.dump(feature_columns, open(f'{data_dir}/feature.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_data, open(f'{data_dir}/train_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, open(f'{data_dir}/test_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f'保存数据')
    print(f'========= Build Model =========')
    train_config = cfg['train']
    model_config = cfg['model']
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = DeepFM(feature_columns, cfg)
        model.summary()
        model.compile(loss=train_config['loss'], optimizer=train_config['optimizer'], metrics=model_config['metrics'])
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    train_history = model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size, validation_split=0.1)
    res = model.evaluate(test_data[0], test_data[1], batch_size=batch_size)
    print(f'test AUC: {res[1]}')

    print('========= Export Model Information =========')
    date = get_date()
    dirs = [__model__, date]
    directory = f'../result'
    for d in dirs:
        directory = os.path.join(directory, d)
        if not os.path.exists(directory):
            os.mkdir(directory)

    export_config(bcfg, directory)
    model.save_weights(f'{directory}/weights.h5')
    cost = time.time() - start
    export_result(train_history, res, directory, cost)
    print(f'========= Train over, cost: {cost:.3f}s =========')


if __name__ == '__main__':
    train(config)
