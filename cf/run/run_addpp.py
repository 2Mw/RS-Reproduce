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
        re_rank_result_sw_5_all = []
        re_rank_result_sw_7_all = []
        re_rank_result_sw_9_all = []
        
        re_rank_result_v_all = []
        re_rank_result_v_sw_5_all = []
        re_rank_result_v_sw_7_all = []
        re_rank_result_v_sw_9_all = []
        
        re_rank_result_mmr_all = []
        
        MRR = 0
        ILAD = 0
        ILMD = 0
        
        MRR_sw_5 = 0
        ILAD_sw_5 = 0
        ILMD_sw_5 = 0

        MRR_sw_7 = 0
        ILAD_sw_7 = 0
        ILMD_sw_7 = 0

        MRR_sw_9 = 0
        ILAD_sw_9 = 0
        ILMD_sw_9 = 0
        
        # vanilla dpp
        MRR_v= 0
        ILAD_v = 0
        ILMD_v = 0
        
        MRR_v_sw_5 = 0
        ILAD_v_sw_5 = 0
        ILMD_v_sw_5 = 0
        
        MRR_v_sw_7 = 0
        ILAD_v_sw_7 = 0
        ILMD_v_sw_7 = 0
        
        MRR_v_sw_9 = 0
        ILAD_v_sw_9 = 0
        ILMD_v_sw_9 = 0
        
        # mmr
        MRR_mmr= 0
        ILAD_mmr = 0
        ILMD_mmr = 0
        
        MRR_record = []
        ILAD_record = []
        ILMD_record = []
        
        MRR_sw_5_record = []
        ILAD_sw_5_record = []
        ILMD_sw_5_record = []
        
        MRR_sw_7_record = []
        ILAD_sw_7_record = []
        ILMD_sw_7_record = []
        
        MRR_sw_9_record = []
        ILAD_sw_9_record = []
        ILMD_sw_9_record = []
        
        # vanilla
        MRR_v_record = []
        ILAD_v_record = []
        ILMD_v_record = []
        
        MRR_v_sw_5_record = []
        ILAD_v_sw_5_record = []
        ILMD_v_sw_5_record = []
        
        MRR_v_sw_7_record = []
        ILAD_v_sw_7_record = []
        ILMD_v_sw_7_record = []
        
        MRR_v_sw_9_record = []
        ILAD_v_sw_9_record = []
        ILMD_v_sw_9_record = []
        
        # mmr
        MRR_mmr_record = []
        ILAD_mmr_record = []
        ILMD_mmr_record = []
        
        def ILAD_fn(x):
            x_diag = np.diag(np.diag(x))
            x = x - x_diag
            return x
        
        def ILMD_fn(x):
            l = x.shape[0]
            inf_diag = np.diagflat([np.inf for _ in range(l)])
            x_diag = np.diag(np.diag(x))
            x = x - x_diag
            x = x + inf_diag
            x[x < 0] = np.inf
            return x
            
        
        # Get 10000 people's interests
        for i in tqdm(range(10000)):
            item_list = user_item_interest_data[i]
            item_relavance_score = user_item_interest_score_data[i]
            item_relavance_score_weight = y_pred[i]
            item_to_item_sim_matrix = item_sim_matrix[item_list, :][:, item_list]
            item_indices = dpp(item_to_item_sim_matrix, 20, item_relavance_score, item_relavance_score_weight)
            vanilla_item_indices = dpp(item_to_item_sim_matrix, 20)
            item_indices_sw_5 = dpp_sw(item_to_item_sim_matrix, 5, 20, item_relavance_score, item_relavance_score_weight)
            item_indices_sw_7 = dpp_sw(item_to_item_sim_matrix, 7, 20, item_relavance_score, item_relavance_score_weight)
            item_indices_sw_9 = dpp_sw(item_to_item_sim_matrix, 9, 20, item_relavance_score, item_relavance_score_weight)
            vanilla_item_indices_sw_5 = dpp_sw(item_to_item_sim_matrix, 5, 20)
            vanilla_item_indices_sw_7 = dpp_sw(item_to_item_sim_matrix, 7, 20)
            vanilla_item_indices_sw_9 = dpp_sw(item_to_item_sim_matrix, 9, 20)
            
            mmr_indices = mmr(item_to_item_sim_matrix, 20, item_relavance_score)
            
            re_rank_result = item_list[item_indices]
            re_rank_result_sw_5 = item_list[item_indices_sw_5]
            re_rank_result_sw_7 = item_list[item_indices_sw_7]
            re_rank_result_sw_9 = item_list[item_indices_sw_9]
            
            re_rank_v_result = item_list[vanilla_item_indices]
            re_rank_v_result_sw_5 = item_list[vanilla_item_indices_sw_5]
            re_rank_v_result_sw_7 = item_list[vanilla_item_indices_sw_7]
            re_rank_v_result_sw_9 = item_list[vanilla_item_indices_sw_9]
            
            
            re_rank_mmr_result = item_list[mmr_indices]
            
            for i, j in enumerate(re_rank_result):
                if j == item_list[0]:
                    MRR += 1 / (i + 1)
                    break
                    
            for i, j in enumerate(re_rank_result_sw_5):
                if j == item_list[0]:
                    MRR_sw_5 += 1 / (i + 1)
                    break
            
            for i, j in enumerate(re_rank_result_sw_7):
                if j == item_list[0]:
                    MRR_sw_7 += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_result_sw_9):
                if j == item_list[0]:
                    MRR_sw_9 += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_v_result):
                if j == item_list[0]:
                    MRR_v += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_v_result_sw_5):
                if j == item_list[0]:
                    MRR_v_sw_5 += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_v_result_sw_7):
                if j == item_list[0]:
                    MRR_v_sw_7 += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_v_result_sw_9):
                if j == item_list[0]:
                    MRR_v_sw_9 += 1 / (i + 1)
                    break
                
            for i, j in enumerate(re_rank_mmr_result):
                if j == item_list[0]:
                    MRR_mmr += 1 / (i + 1)
                    break

            ILAD += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_result, :][:, re_rank_result]))
            ILMD += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_result, :][:, re_rank_result]))
            ILAD_sw_5 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_result_sw_5, :][:, re_rank_result_sw_5]))
            ILMD_sw_5 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_result_sw_5, :][:, re_rank_result_sw_5]))
            ILAD_sw_7 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_result_sw_7, :][:, re_rank_result_sw_7]))
            ILMD_sw_7 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_result_sw_7, :][:, re_rank_result_sw_7]))
            ILAD_sw_9 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_result_sw_9, :][:, re_rank_result_sw_9]))
            ILMD_sw_9 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_result_sw_9, :][:, re_rank_result_sw_9]))
            
            ILAD_v += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_v_result, :][:, re_rank_v_result]))
            ILMD_v += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_v_result, :][:, re_rank_v_result]))
            ILAD_v_sw_5 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_v_result_sw_5, :][:, re_rank_v_result_sw_5]))
            ILMD_v_sw_5 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_v_result_sw_5, :][:, re_rank_v_result_sw_5]))
            ILAD_v_sw_7 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_v_result_sw_7, :][:, re_rank_v_result_sw_7]))
            ILMD_v_sw_7 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_v_result_sw_7, :][:, re_rank_v_result_sw_7]))
            ILAD_v_sw_9 += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_v_result_sw_9, :][:, re_rank_v_result_sw_9]))
            ILMD_v_sw_9 += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_v_result_sw_9, :][:, re_rank_v_result_sw_9]))
            
            ILAD_mmr += np.mean(ILAD_fn(1 - item_sim_matrix[re_rank_mmr_result, :][:, re_rank_mmr_result]))
            ILMD_mmr += np.min(ILMD_fn(1 - item_sim_matrix[re_rank_mmr_result, :][:, re_rank_mmr_result]))
            
            re_rank_result_all.append(re_rank_result)
            re_rank_result_sw_5_all.append(re_rank_result_sw_5)
            re_rank_result_sw_7_all.append(re_rank_result_sw_7)
            re_rank_result_sw_9_all.append(re_rank_result_sw_9)
            
            re_rank_result_v_all.append(re_rank_v_result)
            re_rank_result_v_sw_5_all.append(re_rank_v_result_sw_5)
            re_rank_result_v_sw_7_all.append(re_rank_v_result_sw_7)
            re_rank_result_v_sw_9_all.append(re_rank_v_result_sw_9)
            
            re_rank_result_mmr_all.append(re_rank_mmr_result)
            
            
            MRR_record.append(MRR)
            ILAD_record.append(ILAD)
            ILMD_record.append(ILMD)
            
            MRR_sw_5_record.append(MRR_sw_5)
            ILAD_sw_5_record.append(ILAD_sw_5)
            ILMD_sw_5_record.append(ILMD_sw_5)
            
            MRR_sw_7_record.append(MRR_sw_7)
            ILAD_sw_7_record.append(ILAD_sw_7)
            ILMD_sw_7_record.append(ILMD_sw_7)
            
            MRR_sw_9_record.append(MRR_sw_9)
            ILAD_sw_9_record.append(ILAD_sw_9)
            ILMD_sw_9_record.append(ILMD_sw_9)
            
            MRR_v_record.append(MRR_v)
            ILAD_v_record.append(ILAD_v)
            ILMD_v_record.append(ILMD_v)
            
            MRR_v_sw_5_record.append(MRR_v_sw_5)
            ILAD_v_sw_5_record.append(ILAD_v_sw_5)
            ILMD_v_sw_5_record.append(ILMD_v_sw_5)
            
            MRR_v_sw_7_record.append(MRR_v_sw_7)
            ILAD_v_sw_7_record.append(ILAD_v_sw_7)
            ILMD_v_sw_7_record.append(ILMD_v_sw_7)
            
            MRR_v_sw_9_record.append(MRR_v_sw_9)
            ILAD_v_sw_9_record.append(ILAD_v_sw_9)
            ILMD_v_sw_9_record.append(ILMD_v_sw_9)
            
            MRR_mmr_record.append(MRR_mmr)
            ILAD_mmr_record.append(ILAD_mmr)
            ILMD_mmr_record.append(ILMD_mmr)
            
        addpp_records = [MRR_record, ILAD_record, ILMD_record, MRR_sw_5_record, ILAD_sw_5_record, ILMD_sw_5_record, MRR_sw_7_record, ILAD_sw_7_record, ILMD_sw_7_record, MRR_sw_9_record, ILAD_sw_9_record, ILMD_sw_9_record]
        v_dpp_records = [MRR_v_record, ILAD_v_record, ILMD_v_record, MRR_v_sw_5_record, ILAD_v_sw_5_record, ILMD_v_sw_5_record, MRR_v_sw_7_record, ILAD_v_sw_7_record, ILMD_v_sw_7_record, MRR_v_sw_9_record, ILAD_v_sw_9_record, ILMD_v_sw_9_record]
        mmr_records = [MRR_mmr_record, ILAD_mmr_record, ILMD_mmr_record]
        pickle.dump(addpp_records, open(os.path.join(os.path.dirname(weight), f'addpp_records.pkl'), 'wb'))
        pickle.dump(v_dpp_records, open(os.path.join(os.path.dirname(weight), f'v_dpp_records.pkl'), 'wb'))
        pickle.dump(mmr_records, open(os.path.join(os.path.dirname(weight), f'mmr_records.pkl'), 'wb'))
        
        
        pickle.dump(np.array(re_rank_result_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_sw_5_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_5_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_sw_7_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_7_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_sw_9_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_9_all.pkl'), 'wb'))
        
        pickle.dump(np.array(re_rank_result_v_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_v_sw_5_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_5_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_v_sw_7_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_7_all.pkl'), 'wb'))
        pickle.dump(np.array(re_rank_result_v_sw_9_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_9_all.pkl'), 'wb'))
        
        
        pickle.dump(np.array(re_rank_result_mmr_all), open(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_mmr_all.pkl'), 'wb'))
        
        
        
        pd.DataFrame(np.array(re_rank_result_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_sw_5_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_5_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_sw_7_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_7_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_sw_9_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_sw_9_all.csv'), index=False)
        
        pd.DataFrame(np.array(re_rank_result_v_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_v_sw_5_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_5_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_v_sw_7_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_7_all.csv'), index=False)
        pd.DataFrame(np.array(re_rank_result_v_sw_9_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_v_sw_9_all.csv'), index=False)
        
        pd.DataFrame(np.array(re_rank_result_mmr_all)).to_csv(os.path.join(os.path.dirname(weight), f'pred-re_rank_result_mmr_all.csv'), index=False)
        print("========= AD-DPP ==========")
        print(f'MRR: {MRR}\tILAD: {ILAD}\tILMD: {ILMD}\nMRR_sw_5: {MRR_sw_5}\tILAD_sw_5: {ILAD_sw_5}\t ILMD_sw_5: {ILMD_sw_5}\nMRR_sw_7: {MRR_sw_7}\tILAD_sw_7: {ILAD_sw_7}\t ILMD_sw_7: {ILMD_sw_7}\nMRR_sw_9: {MRR_sw_9}\tILAD_sw_9: {ILAD_sw_9}\t ILMD_sw_9: {ILMD_sw_9}\n')
        print("========= Vanilla DPP ==========")
        print(f'MRR_v: {MRR_v}\tILAD_v: {ILAD_v}\tILMD_v: {ILMD_v}\nMRR_v_sw_5: {MRR_v_sw_5}\tILAD_v_sw_5: {ILAD_v_sw_5}\t ILMD_v_sw_5: {ILMD_v_sw_5}\nMRR_v_sw_7: {MRR_v_sw_7}\tILAD_v_sw_7: {ILAD_v_sw_7}\t ILMD_v_sw_7: {ILMD_v_sw_7}\nMRR_v_sw_9: {MRR_v_sw_9}\tILAD_v_sw_9: {ILAD_v_sw_9}\t ILMD_v_sw_9: {ILMD_v_sw_9}\n')
        print("========= MMR ==========")
        print(f'MRR_mmr: {MRR_mmr}\tILAD_mmr: {ILAD_mmr}\tILMD_mmr: {ILMD_mmr}\n')
    
        
if __name__ == '__main__':
    with open('/data/amax/b510/yl/repo/33/22/rs/cf/tune/addpp/20240313210616/0.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
        # train(config, '/data/amax/b510/yl/repo/33/22/rs/cf/result/addpp/20240314150518/weights.hdf5')
        ## demo predict
        predict(config, '/data/amax/b510/yl/repo/33/22/rs/cf/result/addpp/20240314175553/model.tf')
    
    # a = np.random.randint(1, 10, (5, 5))
    # b = [1, 2]
    # print(a, a[b, :][:, b], a[b, b], sep='\n')