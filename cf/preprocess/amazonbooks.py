import numpy as np

import cf.preprocess.data as base
import pandas as pd
from cf.preprocess.feature_column import *

BEHAVIOR_NAMES = ['UserID', 'ItemID', 'Rating', 'TimeStamp']

SEP = ','


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms',
                   model_type=""):
    if model_type == 'recall':
        if 'ratings_Books.csv' not in file:
            e = f'The train file name must be ratings_Books.csv'
            raise ValueError(e)
        behavior = base.read_data(file, sample_num, SEP, BEHAVIOR_NAMES)

        sub = behavior[['UserID', 'ItemID']]
        sub = sub.drop_duplicates().reset_index(drop=True)

        # 找到交互多的用户，过滤数据较少的用户
        gu = sub.groupby(['UserID'])
        utl = ((gu.count() > 5) & (gu.count() < 500)).reset_index()
        utl.columns = ['UserID', 'USuit']
        # 找到交互多的 item，过滤数据较少的 item
        iu = sub.groupby(['ItemID'])
        itl = (iu.count() > 200).reset_index()
        itl.columns = ['ItemID', 'ISuit']
        # 合并
        behavior = behavior.merge(utl).merge(itl)
        behavior = behavior[behavior['USuit']]
        behavior = behavior[behavior['ISuit']]
        # 处理 ratings
        behavior['LikeType'] = np.where(behavior['Rating'] >= 3, 1, 0)
        # 删除掉负样本，负样本在训练的时候进行拿取
        behavior = behavior[behavior['LikeType'] == 1]
        behavior = behavior.drop(columns=['USuit', 'ISuit', 'Rating', 'TimeStamp', 'LikeType'])
        base.mapped2sequential(behavior, ['UserID', 'ItemID'])
        item_list_per_user = behavior.groupby(['UserID'])['ItemID'].apply(list).reset_index()
        item_list_per_user.columns = ['UserID', 'ItrList']
        item_list_per_user['true_list'] = item_list_per_user['ItrList'].apply(lambda x: True if len(x) > 3 else False)
        item_list_per_user = item_list_per_user[item_list_per_user['true_list']]
        item_list_per_user = item_list_per_user.drop(columns=['true_list'])
        behavior = behavior.merge(item_list_per_user, on='UserID')
        # 抽取训练集和测试集
        behavior = behavior.sample(frac=1).reset_index(drop=True)
        length = behavior.shape[0]
        train_data = behavior[:int(length * (1 - test_size))].reset_index(drop=True)
        test_data = behavior[int(length * (1 - test_size)):].reset_index(drop=True)

        item_data = behavior['ItemID'].drop_duplicates().reset_index(drop=True)
        item_data = pd.DataFrame(item_data)
        # 拼接成可以训练的数据格式
        query_col = ['UserID', 'ItrList']
        tmp = behavior.iloc[0, :][query_col]
        item_df = pd.DataFrame([tmp.tolist()] * item_data.shape[0], columns=query_col)
        item_df = pd.concat([item_df, item_data], axis=1)
        item_df = item_df[train_data.columns]
        mapped_columns = ['C1]]query', 'C2]]item', 'S1]]item']
        train_data.columns = mapped_columns
        test_data.columns = mapped_columns
        item_df.columns = mapped_columns
        fc = {
            SparseFeat('C1]]query', behavior['UserID'].max() + 1, np.int32),
            SparseFeat('C2]]item', behavior['ItemID'].max() + 1, np.int32),
            SequenceFeat('S1]]item', behavior['ItemID'].max() + 1, np.int32),
        }
        train_x = {f.name: train_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        train_x.update({f.name: train_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        query_data = {f.name: test_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        query_data.update({f.name: test_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        item_data = {f.name: item_df[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        item_data.update({f.name: item_df[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        return fc, train_x, query_data, item_data
    else:
        raise NotImplementedError
