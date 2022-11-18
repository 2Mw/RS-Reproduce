import pickle

import numpy as np

import cf.preprocess.data as base
import os
import pandas as pd
from cf.preprocess.feature_column import *

# fliggy dataset: https://tianchi.aliyun.com/dataset/dataDetail?dataId=113649

# ========= user_item_behavior_history.csv ==============
# UserID	    整数类型，序列化后的用户ID
# ItemID	    整数类型，序列化后的商品ID
# BehaviorType	字符串，枚举类型，包括('clk', 'fav', 'cart', 'pay')
# Timestamp     整数类型，行为发生的时间戳

BEHAVIOR_NAMES = ['UserID', 'ItemID', 'BehaviorType', 'TimeStamp']
MAP_BEHAVIOR = {'clk': 0, 'fav': 0, 'cart': 1, 'pay': 1}

# ========= user_profile.csv =================
# 用户ID	整数类型，序列化后的用户ID
# 年龄	整数类型，序列化后的年龄ID
# 性别	整数类型，序列化后性别ID, 1 - man,  2 - woman, 3 - not defined.
# 职业	整数类型，序列化后职业ID, -1 not defined.
# 常居城市	整数类型，序列化后城市ID
# 人群标签	字符串，每个标签序列化后ID用英文分号分割

USER_NAMES = ['UserID', 'Age', 'Gender', 'Occupation', 'UserCity', 'uLabel']
USER_LABELS = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# ============== item_profile.csv ==============
# 商品ID	整数类型，序列化后的商品ID
# 商品类目ID	整数类型，序列化后的商品类目ID
# 商品城市	整数类型，序列化后的商品城市ID
# 商品标签	字符串，每个标签序列化后ID用英文分号分割

ITEM_NAMES = ['ItemID', 'CateID', 'Item_city', 'iLabel']

# ============== All features ======================
# Format: [ <User Feature>, <Item feature>, 'Label' ]
ALL_NAMES = ['UserID', 'Age', 'Gender', 'Occupation', 'UserCity', 'uLabel', 'ItemID', 'CateID', 'ItemCity', 'iLabel',
             'BehaviorType']
MAPPED_NAMES = ['C1', 'I1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'label']

sparse_feature = [f'C{i}' for i in range(1, 10)]
dense_feature = ['I1']

# Other
SEP = ','


# For the mapped_names 'label', `clk, fav` is 0, `cart, pay` is 1.
def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms',
                   model_type=""):
    """
    Create fliggy dataset.


    :param file: file path of fliggy dataset.
    :param sample_num: sample number of dataset, -1 means all chunks.
    :param test_size: test size of dataset.
    :param numeric_process: numeric process of dataset.The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :param model_type: 不同类型表示不同处理方式
    :return: fc, (train_x, train_y), (test_x, test_y), train_x: {'C1': [1,2,3]}
    """
    if model_type == 'ctr':
        if sample_num != -1:
            raise ValueError('Fliggy sample_num must be all(-1).')
        dirname = os.path.dirname(file)
        user_file, item_file = [os.path.join(dirname, i) for i in ['user_profile.csv', 'item_profile.csv']]
        behaviors = base.read_raw_data(file, sample_num, SEP)
        users = base.read_raw_data(user_file, sample_num, SEP)
        items = base.read_raw_data(item_file, sample_num, SEP)
        # process lines
        data = []
        user_dict, item_dict = {}, {}
        for u in users:
            uLabels = u[-1].split(';')
            arr = ['1' if str(l) in uLabels else '0' for l in USER_LABELS]
            u[-1] = int("".join(arr), base=2)
            user_dict[u[0]] = u
        for i in items:
            if len(i) == 0 or i is None:
                continue
            iLabels = i[-1].split(';')
            e = 1
            for l in iLabels:
                e *= int(l)
            i[-1] = str(e)
            item_dict[i[0]] = i

        for b in behaviors:
            entry = []
            if user_dict.get(b[0]) is None or item_dict.get(b[1]) is None:
                continue
            entry.extend(user_dict[b[0]])
            entry.extend(item_dict[b[1]])
            entry.append(MAP_BEHAVIOR.get(b[2]))
            data.append(entry)
        df = pd.DataFrame(data, columns=MAPPED_NAMES)

        df.astype('str')
        df['I1'] = df['I1'].astype('float32')

        df = base.process(df, sparse_feature, dense_feature, numeric_process=numeric_process)

        fc = base.gen_feature_columns(df, sparse_feature, dense_feature)

        return base.split_dataset(df, fc, test_size)
    elif model_type == 'recall':
        # 对于双塔召回模型的处理
        if 'user_item_behavior_history.csv' not in file:
            e = f'The train file name must be user_item_behavior_history.csv'
            raise ValueError(e)
        dirname = os.path.dirname(file)

        # 由于飞猪中的用户和item交互数据大约有两亿条，全部训练不够现实，这里选择只处理 5kw 条

        user_file, item_file = [os.path.join(dirname, i) for i in ['user_profile.csv', 'item_profile.csv']]
        behavior = base.read_data(file, sample_num, SEP, BEHAVIOR_NAMES)
        users = base.read_data(user_file, -1, SEP, USER_NAMES)
        items = base.read_data(item_file, -1, SEP, ITEM_NAMES)
        # 处理用户异常年龄
        avg_age = int(users['Age'].mean())
        users['Age'] = users['Age'].apply(lambda x: x if x <= 75 else avg_age)
        # behavior 数据集含有 UserID	ItemID	BehaviorType 完全一致的数据，进行排除，使用 BehaviorCount 来表示交互的次数
        behavior = behavior.drop(columns=['TimeStamp'])
        behavior['BehaviorCount'] = 0
        behavior = behavior.groupby(['UserID', 'ItemID', 'BehaviorType']).count().reset_index()
        # 直接进行合并
        behavior = behavior.merge(users, on='UserID').merge(items, on='ItemID')
        # 处理稀疏属性
        base.mapped2sequential(behavior,
                               ['UserID', 'ItemID', 'Occupation', 'CateID', 'BehaviorType', ['UserCity', 'Item_city']])
        # 处理多值属性
        uLabel, u_label_vocab = base.multi_value_process(behavior, 'uLabel', ';')
        iLabel, i_label_vocab = base.multi_value_process(behavior, 'iLabel', ';')
        behavior['uLabels'] = uLabel
        behavior['iLabels'] = iLabel
        behavior = behavior.drop(columns=['uLabel', 'iLabel'])
        # 获取每个用户所有交互过的 item
        item_list_per_user = behavior.groupby(['UserID'])['ItemID'].apply(list).reset_index()
        item_list_per_user.columns = ['UserID', 'InteractItems']
        behavior = behavior.merge(item_list_per_user, on='UserID')
        # 随机打乱顺序
        behavior = behavior.sample(frac=1).reset_index(drop=True)
        length = behavior.shape[0]
        # 抽取训练集和测试集
        train_data = behavior[:int(length * (1 - test_size))].reset_index(drop=True)
        test_data = behavior[int(length * (1 - test_size)):].reset_index(drop=True)
        # 抽取 item 数据
        item_data = behavior[['ItemID', 'CateID', 'Item_city', 'iLabels']]
        # 将 iLabel 列转为字符串方便进行去重处理
        t = item_data['iLabels'].apply(lambda x: ','.join(list(map(str, x))))
        item_data = item_data.drop(columns=['iLabels'])
        item_data['iLabels'] = t
        item_data = item_data.drop_duplicates().reset_index(drop=True)
        # 将字符串转为list
        t = item_data['iLabels'].apply(lambda x: [int(i) for i in x.split(',')])
        item_data = item_data.drop(columns=['iLabels'])
        item_data['iLabels'] = t
        # 拼接成可以训练的数据格式
        query_col = USER_NAMES + ['BehaviorType', 'BehaviorCount', 'InteractItems']
        query_col[5] = 'uLabels'
        tmp = behavior.iloc[0, :][query_col]
        item_df = pd.DataFrame([tmp.tolist()] * item_data.shape[0], columns=query_col)
        item_df = pd.concat([item_df, item_data], axis=1)
        # 整理列的顺序
        item_df = item_df[train_data.columns]
        # 构建 feature columns
        # ['UserID', 'ItemID', 'BehaviorType', 'BehaviorCount', 'Age', 'Gender',  'Occupation', 'UserCity', 'CateID',
        #  'Item_city', 'uLabels', 'iLabels', 'InteractItems']
        mapped_columns = ['C1]]query', 'C2]]item', 'I1', 'I2', 'I3', 'I4', 'C3', 'C4]]city', 'C5', 'C6]]city', 'S1',
                          'S2', 'S3]]item']
        train_data.columns = mapped_columns
        test_data.columns = mapped_columns
        item_df.columns = mapped_columns
        city_max = max(behavior['UserCity'].max(), behavior['Item_city'].max())
        fc = {
            SparseFeat('C1]]query', behavior['UserID'].max() + 1, np.int32),
            SparseFeat('C2]]item', behavior['ItemID'].max() + 1, np.int32),
            DenseFeat('I1', 1, np.float32),
            DenseFeat('I2', 1, np.float32),
            DenseFeat('I3', 1, np.float32),
            DenseFeat('I4', 1, np.float32),
            SparseFeat('C3', behavior['Occupation'].max() + 1, np.int32),
            SparseFeat('C4]]city', city_max + 1, np.int32),
            SparseFeat('C5', behavior['CateID'].max() + 1, np.int32),
            SparseFeat('C6]]city', city_max + 1, np.int32),
            SequenceFeat('S1', u_label_vocab + 1, np.int32),
            SequenceFeat('S2', i_label_vocab + 1, np.int32),
            SequenceFeat('S3]]item', behavior['ItemID'].max() + 1, np.int32),
        }
        train_x = {f.name: train_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        train_x.update({f.name: train_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        query_data = {f.name: test_data[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        query_data.update({f.name: test_data[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        item_data = {f.name: item_df[f.name].values.astype(f.dtype) for f in fc if not isinstance(f, SequenceFeat)}
        item_data.update({f.name: item_df[f.name].values for f in fc if isinstance(f, SequenceFeat)})
        return fc, train_x, query_data, item_data
    else:
        raise NotImplementedError('未知的处理方式')

if __name__ == '__main__':
    f = open(r'E:\Notes\DeepLearning\practice\rs\data\fliggy\recall_data_all\feature.pkl', 'rb')
    print(pickle.load(f))