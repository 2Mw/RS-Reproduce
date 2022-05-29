import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, FunctionTransformer
from cf.utils.logger import logger

# IXX 表示数值型数据, CXX表示类别型数据
NAMES = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3',
         'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
         'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']


def create_criteo_dataset(file: str, sample_num: int = -1, test_size: float = 0.2):
    # 注意力机制需要将每个特征转换为相同维度的embedding, 其他的不需要把数值型特征变换
    """
    使用 sklearn 来对 criteo 进行处理，

    :param file: 文件路径
    :param embedding_dim: embedding 维度
    :param test_size: 测试集比例 (0-1)
    :return: feature_columns, (train_x, train_y)
    """
    if test_size >= 1 or test_size < 0:
        raise ValueError(f'The test_size must in the range of (0,1), but your is {test_size}')

    df = pd.read_csv(file, sep='\t', iterator=True, names=NAMES)
    if sample_num > 0:
        df = df.get_chunk(sample_num)
    else:
        df = df.get_chunk()
    sparse_features = [f'C{i}' for i in range(1, 27)]  # 分类型数据
    dense_features = [f'I{i}' for i in range(1, 14)]  # 数值型数据
    features = sparse_features + dense_features

    logger.info("填充无效值")
    df[sparse_features] = df[sparse_features].fillna('-1')
    df[dense_features] = df[dense_features].fillna(0)

    logger.info("=== Process numeric feature ===")  # 使用 log-normalize or MinMaxScaler
    # est = KBinsDiscretizer(1000, encode='ordinal', strategy='uniform')  # 使用 ordinal 编码而不是 one-hot
    # df[dense_features] = est.fit_transform(df[dense_features])  # 对每一列进行数值离散化处理
    for f in dense_features:
        if f == 'I2':
            df[f] = np.log(df[f] + 4)
        else:
            df[f] = np.log(df[f] + 1)

    logger.info("=== Process categorical feature ===")
    for feature in sparse_features:  # 对于分类型数据进行处理，将对应的类别型数据转为唯一的数字编号
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])  # 输入的数据必须是一维向量

    feature_columns = [
        {'name': feature, 'feature_num': int(df[feature].max()) + 1}
        for feature in features
    ]

    if test_size > 0:
        train, test = train_test_split(df, test_size=test_size)
        del df
        gc.collect()
        train_x = {feature: train[feature].values.astype('float32') for feature in features}
        train_y = train['label'].values.astype('float32')
        test_x = {feature: test[feature].values.astype('float32') for feature in features}
        test_y = test['label'].values.astype('float32')
        return feature_columns, (train_x, train_y), (test_x, test_y)
    else:
        train_x = {feature: df[feature].values.astype('float32') for feature in features}
        train_y = df['label'].values.astype('float32')
        return feature_columns, (train_x, train_y)
