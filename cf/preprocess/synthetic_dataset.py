import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import os
import pickle
import cf

"""
This is synthetic_dataset for re-rank algorithms.
"""

user_cnt, user_vec_dim, user_catelog_cnt = 1000000, 15, 29
item_cnt, item_vec_dim, item_catelog_cnt = 30000, 12, 19

project_dir = cf.get_project_path()

path_prefix = os.path.abspath(os.path.join(project_dir, 'data', 'synthetic'))

"""
Generate systhetic dataset:
* user vector matrix
* item vector matrix
* item similarity matrix
* user -> item interest matrix (user_cnt, 30)
"""
def gen_data():
    # check path
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    
    ## generate item vectors
    item_slot = [[] for i in range(item_catelog_cnt)]
    rng = np.random.default_rng()
    for _ in tqdm(range(item_cnt)):
        slot_idx = random.randint(0, item_catelog_cnt - 1)
        if len(item_slot[slot_idx]) == 0:
            item_slot[slot_idx].append(rng.random(item_vec_dim))
        else:
            # get a vector which cosine_similarity <= cos 20 degree with first vector.
            new_vec = rng.random(item_vec_dim)
            #print(new_vec)
            while cos_sim(new_vec, item_slot[slot_idx][0]) > np.cos(20 * np.pi / 180):
                new_vec = rng.random(item_vec_dim)
            item_slot[slot_idx].append(new_vec)
    
    
    items = []
    for i in item_slot:
        items.extend(i)
        
    # pd.DataFrame(items).to_csv(os.path.join(path_prefix, 'item.csv'))
    pickle.dump(np.array(items), open(os.path.join(path_prefix, 'item.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    items = np.array(items)
    # print(pd.DataFrame(items))
    ## generate item similarity matrix
    item_sim_matrix = np.zeros([item_cnt, item_cnt])
    item_sim_matrix = cosine_similarity(items, items)
    print(cos_sim(items[0], items[1]), item_sim_matrix[0][1])
    # for i in tqdm(range(0, item_cnt)):
    #     for j in range(i, item_cnt):
    #         item_sim_matrix[i][j] = item_sim_matrix[j][i] = cos_sim(items[i], items[j])
            
    #print(pd.DataFrame(item_sim_matrix))
    pickle.dump(np.array(item_sim_matrix), open(os.path.join(path_prefix, 'item_sim_matrix.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    
    ## generate user vectors
    user_slot = [[] for i in range(user_catelog_cnt)]
    rng = np.random.default_rng()
    for _ in tqdm(range(user_cnt)):
        slot_idx = random.randint(0, user_catelog_cnt - 1)
        if len(user_slot[slot_idx]) == 0:
            user_slot[slot_idx].append(rng.random(user_vec_dim))
        else:
            new_vec = rng.random(user_vec_dim)
            while cos_sim(new_vec, user_slot[slot_idx][0]) > np.cos(15 * np.pi / 180):
                new_vec = rng.random(user_vec_dim)
            user_slot[slot_idx].append(new_vec)
    
    users = []
    for i in user_slot:
        users.extend(i)
    
    # pd.DataFrame(users).to_csv(os.path.join(path_prefix, 'user.csv'))
    pickle.dump(np.array(users), open(os.path.join(path_prefix, 'users.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
            
    ## generate user -> item interest matrix
    uim = np.random.randint(item_cnt, size=(user_cnt, 30))
    
    
    uim_score = np.abs(np.random.randn(user_cnt, 30) * 5)
    
    uim_score = (uim_score - np.min(uim_score)) / (np.max(uim_score) - np.min(uim_score))
    
    sorted_indices = np.fliplr(np.argsort(uim_score, axis=1))
    uim = np.array([row[indices] for row, indices in zip(uim, sorted_indices)])
    uim_score = np.array([row[indices] for row, indices in zip(uim_score, sorted_indices)])
    pd.DataFrame(uim).to_csv(os.path.join(path_prefix, 'user-item-interest.csv'))
    pickle.dump(np.array(uim), open(os.path.join(path_prefix, 'user-item-interest.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(uim_score).to_csv(os.path.join(path_prefix, 'user-item-interest-score.csv'))
    pickle.dump(np.array(uim_score), open(os.path.join(path_prefix, 'user-item-interest-score.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    
    
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
if __name__ == '__main__':
    gen_data()
    # a = np.array([[1,2],[2,1],[3,1]])
    # b = np.array([[1,2],[2,1],[3,1]])
    # print(cosine_similarity(a, b))