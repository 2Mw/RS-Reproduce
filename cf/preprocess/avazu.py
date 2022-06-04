import cf.preprocess.data as base

sparse_features = [f'C{i}' for i in range(1, 22)]
dense_features = []
NAMES = ['id', 'label', 'hour'] + sparse_features + dense_features


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

    df = base.read_data(file, sample_num, ',', NAMES)

    df = df.iloc[1:, :]

    df = df.astype('str')

    df = base.process(df, sparse_features, dense_features, numeric_process)

    fc = base.gen_feature_columns(df, sparse_features, dense_features)

    return base.split_dataset(df, fc, test_size)
