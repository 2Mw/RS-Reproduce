import copy
import os.path

import pandas as pd
import tensorflow as tf
from cf.config.addpp import config
from cf.models.rerank.addpp import *
from cf.utils.config import *
import cf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from cf.utils.callbacks import AbnormalAUC, MetricsMonitor
import cf.run.base as base
from cf.preprocess import data as dataloader
from cf.utils.logger import logger
from keras.preprocessing.sequence import pad_sequences as ps
import pickle
import numpy as np
import cf.metric as metric
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = cf.get_project_path()

__model__ = 'addpp'
_RUN_EAGERLY = False
project_dir = cf.get_project_path()

def train(cfg, dataset: str = '', weights:str = '', *args, **kwargs):
    bcfg = copy.deepcopy(cfg)
    start = time.time()
    logger.info(f'========= Loading configures of {__model__} =========')
    train_config = cfg['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    test_batch_size = train_config['test_batch_size']
    opt = train_config['optimizer']
    lr = train_config['lr']
    
    user_data = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'users.pkl'), 'rb'))
    
    user_data_x = user_data[:, :-1]
    user_data_y = user_data[:,-1]
    # user_data_y2 = 1 - user_data[:,-1]
    # user_data_y = np.stack([user_data_y1, user_data_y2], axis=1)
    
    directory = base.create_result_dir(__model__, project_dir)
    logger.info(f'========= Loading over =========')
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = ADDPP(cfg)
        optimizer = keras.optimizers.get(opt)
        optimizer.learning_rate = lr
        loss = tf.metrics.MeanAbsoluteError()
        model.compile(optimizer=optimizer, run_eagerly=_RUN_EAGERLY, metrics=['mae'], loss='mse')
    if os.path.exists(weights):
        model.built = True
        model.load_weights(weights)
        logger.info(f'========= Loading weights of {weights} =========')
    else:
        ckpt = ModelCheckpoint(os.path.join(directory, 'weights.{epoch:03d}.hdf5'), save_weights_only=True)
        model.fit(user_data_x, user_data_y, batch_size=batch_size, callbacks=[ckpt], epochs=epochs)
        model.save_weights(os.path.join(directory, 'weights.hdf5'))
        model.save(os.path.join(directory, 'model.tf'))
        y_pred = model.predict(user_data_x)
        pickle.dump(y_pred, open(os.path.join(directory, f'y_pred.pkl'), 'wb'))
        pd.DataFrame(y_pred).to_csv(os.path.join(directory, f'y_pred.csv'), index=False, header=['y_pred'])
        
def evaluate(cfg, weight: str, *args, **kwargs):
    pass

# 仅支持整个模型的加载
def predict(cfg, weight: str, *args, **kwargs):
    if weight is None or len(weight) == 0:
        raise ValueError('weight not specified.')
    else:
        logger.info(f'========= Loading weights of {weight} =========')
    train_config = cfg['train']
    directory = base.create_result_dir(__model__, project_dir)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    user_data = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'users.pkl'), 'rb'))
    item_data = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'item.pkl'), 'rb'))
    item_sim_matrix = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'item_sim_matrix.pkl'), 'rb'))
    user_item_interest_data = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'user-item-interest.pkl'), 'rb'))
    user_item_interest_score_data = pickle.load(open(os.path.join(project_dir, 'data', 'synthetic', 'user-item-interest-score.pkl'), 'rb'))
    
    user_data_x = user_data[:, :-1]
    with mirrored_strategy.scope():
        model: ADDPP = keras.models.load_model(weight)
        # model.built = True
        # model.load_weights(weight)
        filename = os.path.join(os.path.dirname(weight), f'pred-{os.path.basename(weight)}.csv')
        y_pred = model.predict(user_data_x)
        pickle.dump(y_pred, open(os.path.join(os.path.dirname(weight), f'pred-{os.path.basename(weight)}.pkl'), 'wb'))
        pd.DataFrame(y_pred).to_csv(filename, index=False, header=['y_pred'])
        
        re_rank_result_all = []
        re_rank_result_sw_all = []
        
        MRR = 0
        ILAD = 0
        ILMD = 0
        
        MRR_sw = 0
        ILAD_sw = 0
        ILMD_sw = 0
        
        MRR_record = []
        ILAD_record = []
        ILMD_record = []
        
        MRR_sw_record = []
        ILAD_sw_record = []
        ILMD_sw_record = []
        
        # Get 10000 people's interests
        for i in tqdm(range(10000)):
            item_list = user_item_interest_data[i]
            item_relavance_score = user_item_interest_score_data[i]
            item_relavance_score_weight = y_pred[i]
            item_to_item_sim_matrix = item_sim_matrix[item_list, :][:, item_list]
            item_indices = dpp(item_to_item_sim_matrix, 20, item_relavance_score, item_relavance_score_weight)
            item_indices_sw = dpp_sw(item_to_item_sim_matrix, 7, 20, item_relavance_score, item_relavance_score_weight)
            re_rank_result = item_list[item_indices]
            re_rank_result_sw = item_list[item_indices_sw]
            
            for i, j in enumerate(re_rank_result):
                if j == item_list[0]:
                    MRR += 1 / (i + 1)
                    break
                    
            for i, j in enumerate(re_rank_result_sw):
                if j == item_list[0]:
                    MRR_sw += 1 / (i + 1)
                    break
                
            ILAD += np.mean(1 - item_sim_matrix[re_rank_result, :][:, re_rank_result])
            ILMD += np.min(1 - item_sim_matrix[re_rank_result, :][:, re_rank_result])
            ILAD_sw += np.mean(1 - item_sim_matrix[re_rank_result_sw, :][:, re_rank_result_sw])
            ILMD_sw += np.min(1 - item_sim_matrix[re_rank_result_sw, :][:, re_rank_result_sw])
            
            re_rank_result_all.append(re_rank_result)
            re_rank_result_sw_all.append(re_rank_result_sw)
            
            MRR_record.append(MRR)
            ILAD_record.append(ILAD)
            ILMD_record.append(ILMD)
            
            MRR_sw_record.append(MRR_sw)
            ILAD_sw_record.append(ILAD_sw)
            ILMD_sw_record.append(ILMD_sw)
            
        records = [MRR_record, ILAD_record, ILMD_record, MRR_sw_record, ILAD_sw_record, ILMD_sw_record]
        pickle.dump(records, open(os.path.join(os.path.dirname(weight), f'records.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_sw_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_all.pkl'), 'wb'))
        pd.DataFrame(np.array(re_rank_result_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_sw_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_all.csv'), index=False)
        print(f'MRR: {MRR}\tILAD: {ILAD}\tILMD: {ILMD}\nMRR_sw: {MRR_sw}\tILAD: {ILAD_sw}\t ILMD: {ILMD_sw}')
        
    
        
if __name__ == '__main__':
    with open('/data/amax/b510/yl/repo/33/22/rs/cf/tune/addpp/20240313210616/0.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
        # train(config, '/data/amax/b510/yl/repo/33/22/rs/cf/result/addpp/20240314150518/weights.hdf5')
        ## demo predict
        predict(config, '/data/amax/b510/yl/repo/33/22/rs/cf/result/addpp/20240314175553/model.tf')
    
    # a = np.random.randint(1, 10, (5, 5))
    # b = [1, 2]
    # print(a, a[b, :][:, b], a[b, b], sep='\n')