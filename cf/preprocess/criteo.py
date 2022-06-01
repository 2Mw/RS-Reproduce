import gc
from sklearn.model_selection import train_test_split
import cf.preprocess.data as base

# IXX 表示数值型数据, CXX表示类别型数据
dense_features = [f'I{i}' for i in range(1, 14)]  # 数值型数据
sparse_features = [f'C{i}' for i in range(1, 27)]  # 分类型数据
label = ['label']
NAMES = label + dense_features + sparse_features


def create_dataset(file: str, sample_num: int = -1, test_size: float = 0.2, numeric_process: str = 'mms'):
    # 注意力机制需要将每个特征转换为相同维度的embedding, 其他的不需要把数值型特征变换
    """
    使用 sklearn 来对 criteo 进行处理，

    :param sample_num: The sample size.
    :param file: 文件路径
    :param test_size: 测试集比例 (0-1)
    :param numeric_process: The way of processing numerical feature ln-LogNormalize, kbd-KBinsDiscretizer, mms-MinMaxScaler
    :return: fc, (train_x, train_y), (test_x, test_y), train_x: {'C1': [1,2,3]}
    """
    if test_size >= 1 or test_size < 0:
        raise ValueError(f'The test_size must in the range of (0,1), but your is {test_size}')

    df = base.read_data(file, sample_num, '\t', NAMES)

    df = base.process(df, sparse_features, dense_features, numeric_process)

    fc = base.gen_feature_columns(df, sparse_features, dense_features)

    if test_size > 0:
        train, test = train_test_split(df, test_size=test_size)
        del df
        gc.collect()
        train_x = {f.name: train[f.name].values.astype(f.dtype) for f in fc}
        train_y = train['label'].values.astype('int32')
        test_x = {f.name: test[f.name].values.astype(f.dtype) for f in fc}
        test_y = test['label'].values.astype('int32')
        return fc, (train_x, train_y), (test_x, test_y)
    else:
        train_x = {f.name: df[f.name].values.astype(f.dtype) for f in fc}
        train_y = df['label'].values.astype('float32')
        return fc, (train_x, train_y)
