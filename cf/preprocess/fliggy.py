import cf.preprocess.data as base
import os
import pandas as pd

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
def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    """
    Create fliggy dataset.

    :param file: file path of fliggy dataset.
    :param sample_num: sample number of dataset, -1 means all chunks.
    :param test_size: test size of dataset.
    :param numeric_process: numeric process of dataset.The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :return: fc, (train_x, train_y), (test_x, test_y), train_x: {'C1': [1,2,3]}
    """
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

    df = base.process(df, sparse_feature, dense_feature, numeric_process)

    fc = base.gen_feature_columns(df, sparse_feature, dense_feature)

    return base.split_dataset(df, fc, test_size)
