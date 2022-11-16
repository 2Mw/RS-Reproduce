import time
import pickle
import numpy as np
import pandas as pd
import os
from cf.utils.logger import logger
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, LabelEncoder
from cf.preprocess import criteo, datasets, movielens, avazu, tbadclick, fliggy, huawei, ml100k
from cf.preprocess.feature_column import DenseFeat, SparseFeat, SequenceFeat
from sklearn.model_selection import train_test_split
import tensorflow as tf

# consts
_pickle = 'pickle'
_feather = 'feather'

numeric_process_way = ['ln', 'kbd', 'mms']
data_types = [_pickle, _feather]


def read_data(file: str, sample_size, sep, names=None, dtype=None):
    """
    Read dataset from files by pandas.

    :param sep: The delimiter of row items.
    :param file: The filepath of source.
    :param sample_size: The number of rows you want to read, -1 means all.
    :param names: The columns name.
    :param dtype: The type of columns.
    :return:
    """
    if not os.path.exists(file):
        e = f'The file: {file} not exists.'
        logger.error(e)
        raise FileNotFoundError(e)
    df = pd.read_csv(file, iterator=True, names=names, sep=sep, dtype=dtype)
    if sample_size > 0:
        df = df.get_chunk(sample_size)
    else:
        df = df.get_chunk()
    return df


def read_raw_data(file: str, sample_num, sep: str):
    """
    Read file by builtin function.

    :param file: The filepath of source.
    :param sample_num: The number of rows you want to read, -1 means all.
    :param sep: The delimiter of row items.
    :return: a group seperated by `sep`
    """
    if not os.path.exists(file):
        e = f'File:{file} not exists'
        logger.error(e)
        raise FileNotFoundError(e)
    content = []
    length = 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = []
        if sample_num > 0:
            while length < sample_num:
                lines.append(f.readline())
                length += 1
        else:
            lines = f.readlines()
        for line in lines:
            content.append(line.strip().split(sep))
    return content


def process(df: pd.DataFrame, sparse_features, dense_features, sequence_features=None, numeric_process: str = 'mms',
            numeric_fn=None):
    """
    Process sparse features, dense features and sequence_features.

    Process sparse feature with LabelEncode().

    Process dense feature with KBinsDiscretizer / LogNormalize / MinMaxScaler

    Process sequence feature with nothing

    :param df: The data frame of pandas.
    :param sparse_features: Sparse features columns name.
    :param dense_features: Dense features columns name.
    :param sequence_features: Sparse features with multi-values
    :param numeric_process: The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :param numeric_fn: The customized numeric process function
    :return:
    """
    if sequence_features is None:
        sequence_features = []
    s = time.time()
    logger.info("=== Padding NAN value ===")
    df[sparse_features] = df[sparse_features].fillna('-1')
    df[dense_features] = df[dense_features].fillna(0)
    if sequence_features is not None:
        df[sequence_features] = df[sequence_features].fillna('-1')
    if len(dense_features) > 0:
        logger.info("=== Process numeric feature ===")
        numeric_process = numeric_process.lower()
        if numeric_process not in numeric_process_way:
            e = f'Wrong way of processing numerical feature: {numeric_process}'
            logger.error(e)
            raise ValueError(e)
        if numeric_process == 'kbd':
            est = KBinsDiscretizer(1000, encode='ordinal', strategy='uniform')  # 使用 ordinal 编码而不是 one-hot
            df[dense_features] = est.fit_transform(df[dense_features])  # 对每一列进行数值离散化处理
        elif numeric_process == 'ln':
            # The way of processing numerical feature in DCNv2
            if numeric_fn is None:
                for f in dense_features:
                    df[f] = np.log(df[f])
            else:
                # Use the way of customized numeric processing.
                numeric_fn(df, dense_features)
        elif numeric_process == 'mms':
            # The way of processing numerical feature in cowclip
            mms = MinMaxScaler()  # scale all values in (0,1)
            df[dense_features] = mms.fit_transform(df[dense_features])
        else:
            e = f'The way {numeric_process} of process numeric features is not supported.'
            logger.error(e)
            raise NotImplementedError(e)
    logger.info("=== Process categorical feature ===")
    for feature in sparse_features:  # 对于分类型数据进行处理，将对应的类别型数据转为唯一的数字编号
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])  # 输入的数据必须是一维向量
    for feature in sequence_features:
        # Do nothing
        pass
    logger.info(f'=== Process data over, cost: {time.time() - s:.2f} seconds. ===')
    return df


def gen_feature_columns(data, sparse_features, dense_features, sequence_features=None, seq_map=None):
    """
    Generate a list of feature columns as format [SparseFeat, ..., DenseFeat, ..., SequenceFeat]

    SparseFeat format like namedtuple('SparseFeat', ['name', 'vocab_size', 'dtype'])

    DenseFeat = namedtuple('DenseFeat', ['name', 'dim', 'dtype'])

    SequenceFeat = namedtuple('SequenceFeat', ['name', 'vocab_size', 'dtype'])

    **对于共享 Embedding 的 feature, 应该采用 "xxx::key"的形式**，比如 C1::item

    :param data: data
    :param sparse_features: The names of sparse features.
    :param dense_features: The names of dense features.
    :param sequence_features: The names of sequence features.
    :param seq_map: if sequence_features is not None, you should assign seq seq_division, else the format like:
    {'S1': 556}, the number is vocabulary size
    :return: [SparseFeat, ..., DenseFeat, ..., SequenceFeat]
    """
    # sparse = [SparseFeat(f, data[f].max()+1) for f in sparse_features]
    # dense = [DenseFeat(feat, 1, None) for feat in dense_features]
    if sequence_features is None:
        sequence_features = []
    else:
        if seq_map is None or len(seq_map) == 0:
            e = 'if sequence_features is not None, you should assign seq seq_division'
            logger.error(e)
            raise ValueError(e)
    sparse = [SparseFeat(name, data[name].max() + 1, np.int32) for name in sparse_features]
    dense = [DenseFeat(name, 1, np.float32) for name in dense_features]
    seq = []
    # 这里不应该扫描全表了，应该直接使用 seq_map 变量
    for name in sequence_features:
        vocab = seq_map[name] + 1
        seq.append(SequenceFeat(name, vocab, data[name].dtype))
    return sparse + dense + seq


def load_data(dataset: str, base: str, sample_size: int, test_ratio: float, train_file, data_type='pickle',
              num_process: str = 'mms', prefix: str = ""):
    """
    Load feature columns, train data, test data from files.

    :param dataset: The name of dataset
    :param base: The base filepath of dataset.
    :param sample_size: The size of train data, if is -1 means all data.
    :param test_ratio: The ratio of test data.
    :param train_file: The name of train data file.
    :param data_type: data store type, feather or pickle
    :param num_process: The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :param prefix: the prefix for saving files' path name, such as 'ctr' or 'recall'
    :return: feature_columns, train_data, test_data
    """
    dataset = dataset.lower()
    if dataset not in datasets:
        e = f'Not supported dataset {dataset} except {datasets}'
        logger.error(e)
        raise ValueError(e)

    if data_type not in data_types:
        e = f'Not supported data store type {data_type} except {data_types}'
        logger.error(e)
        raise ValueError(e)

    if len(prefix) > 0:
        prefix = prefix.lower()
        if prefix in ['ctr', 'recall']:
            if prefix == 'ctr':
                prefix = ''
            elif prefix == 'recall':
                prefix = prefix + '_'
    if sample_size == -1:
        data_dir = os.path.join(base, f'{prefix}data_all')
    else:
        data_dir = os.path.join(base, f'{prefix}data_{sample_size}')

    # concat the name of different type of data
    data_type = data_type.lower()
    files = ['feature', 'train_data', 'test_data', 'item_data']
    suffix = 'feature' if data_type == _feather else 'pkl'
    files = [f'{file}.{suffix}' for file in files]

    # check if exists
    all_exists = True
    if os.path.exists(data_dir):
        for file in files:
            if not os.path.exists(os.path.join(data_dir, file)):
                all_exists = False
                break
    else:
        all_exists = False
        os.mkdir(data_dir)

    fc, train_data, test_data = None, None, None
    if all_exists:
        # files all exist
        logger.info(f'=== Read stored data ===')
        if data_type == _pickle:
            fc = pickle.load(open(f'{data_dir}/{files[0]}', 'rb'))
            train_data = pickle.load(open(f'{data_dir}/{files[1]}', 'rb'))
            test_data = pickle.load(open(f'{data_dir}/{files[2]}', 'rb'))
        elif data_type == _feather:
            e = 'Feather type current not supported.'
            logger.error(e)
            raise NotImplementedError(e)
    else:
        logger.info(f'=== Start to preprocess {dataset} data ===')
        prefix = 'ctr' if prefix == '' else prefix
        if dataset == 'criteo':
            fc, train_data, test_data = criteo.create_dataset(train_file, sample_size, test_ratio, num_process, prefix)
        elif dataset == 'ml':
            fc, train_data, test_data = movielens.create_dataset(train_file, sample_size, test_ratio, num_process,
                                                                 prefix)
        elif dataset == 'avazu':
            fc, train_data, test_data = avazu.create_dataset(train_file, sample_size, test_ratio, num_process, prefix)
        elif dataset == 'tbadclick':
            fc, train_data, test_data = tbadclick.create_dataset(train_file, sample_size, test_ratio, num_process,
                                                                 prefix)
        elif dataset == 'fliggy':
            fc, train_data, test_data = fliggy.create_dataset(train_file, sample_size, test_ratio, num_process, prefix)
        elif dataset == 'huawei':
            fc, train_data, test_data = huawei.create_dataset(train_file, sample_size, test_ratio, num_process, prefix)
        elif dataset == 'ml100k':
            fc, train_data, test_data, items = ml100k.create_dataset(train_file, sample_size, test_ratio, num_process,
                                                                     prefix)
            pickle.dump(items, open(f'{data_dir}/{files[3]}', 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f'Not implement this dataset:{dataset}')
        # read data over, then dump to file.
        logger.info(f'=== dump data ===')
        if data_type == _pickle:
            pickle.dump(fc, open(f'{data_dir}/{files[0]}', 'wb'), pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_data, open(f'{data_dir}/{files[1]}', 'wb'), pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_data, open(f'{data_dir}/{files[2]}', 'wb'), pickle.HIGHEST_PROTOCOL)
        elif data_type == _feather:
            e = 'Dumping the type of feather files currently not supported.'
            logger.error(e)
            raise NotImplementedError(e)
    return fc, train_data, test_data


def split_dataset(df, fc, test_size):
    """
    Split dataset to train and test dataset according to test_size.

    :param df: The dataframe of dataset.
    :param fc: The feature columns of dataset.
    :param test_size: The ratio of test data.
    :return: if test_size > 0 return fc, (train_x, train_y), (test_x, test_y); if test_size = 0 return fc, (train_x, train_y)
    """
    if test_size > 0:
        train, test = train_test_split(df, test_size=test_size)
        train_x = {f.name: train[f.name].values.astype(f.dtype) for f in fc}
        train_y = train['label'].values.astype('int32')
        test_x = {f.name: test[f.name].values.astype(f.dtype) for f in fc}
        test_y = test['label'].values.astype('int32')
        return fc, (train_x, train_y), (test_x, test_y)
    else:
        train_x = {f.name: df[f.name].values.astype(f.dtype) for f in fc}
        if df.get('label') is not None:
            train_y = df['label'].values.astype('int32')
            return fc, (train_x, train_y)
        else:
            return fc, (train_x,)


def mapped2sequential(df: pd.DataFrame, columns: list, start_from_1=True):
    """
    Map the discrete value to sequential id.

    支持共享映射:
    e.g.    1 -- columns = ['a', 'b', 'c']
            2 -- columns = [['a', 'b'], 'c']  表示 'a', 'b' 共用一份 map

    :param df: pd.DataFrame
    :param columns: the columns you want to process with `mapped2sequential`
    :param start_from_1: id 是否從一開始計算，防止在使用 mask_zero 的時候被 mask 掉
    :return:
    """
    for c in columns:
        m = {}
        sign = 1 if start_from_1 else 0
        if isinstance(c, str):
            if c in df.columns:
                l = df[c].unique().tolist()
                for v in l:
                    if m.setdefault(v, sign) == sign:
                        sign += 1
                df[c] = df[c].map(m)
        elif isinstance(c, list):
            for sub_c in c:
                if isinstance(sub_c, str):
                    if sub_c in df.columns:
                        l = df[sub_c].unique().tolist()
                        for v in l:
                            if m.setdefault(v, sign) == sign:
                                sign += 1
                        df[sub_c] = df[sub_c].map(m)
                else:
                    raise ValueError('最多支持二级list')


def min_max_normalize(df: pd.DataFrame, columns: list):
    """
    Add MinMaxNormalize to numeric value.

    :param df:
    :param columns: the columns you want to process with `min_max_normalize`
    :return:
    """
    for c in columns:
        if c in df.columns:
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())


def std_normalize(df: pd.DataFrame, columns: list):
    """
    Add MinMaxNormalize to numeric value.

    :param df:
    :param columns: columns: the columns you want to process with `std_normalize`
    :return:
    """
    for c in columns:
        if c in df.columns:
            df[c] = (df[c] - df[c].mean()) / df[c].std()


def multi_value_process(df: pd.DataFrame, column: str, sep: str):
    """
    对多值属性进行处理，返回处理后的结果以及对应的 vocab_size

    :param df:
    :param column:
    :param sep: 属性分割字符串
    :return: 返回处理后的结果以及对应的 vocab_size
    """
    uMap, ans = {}, []
    sign = 1
    for arr in df[column]:
        que = []
        for i in arr.split(sep):
            if sign == uMap.setdefault(i, sign):
                sign += 1
            que.append(uMap[i])
        ans.append(que)
    return ans, len(uMap) + 1
