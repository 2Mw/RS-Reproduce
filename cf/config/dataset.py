ds_config = {
    'files': {
        # Criteo
        'criteo_base': 'data/criteo',
        'criteo_train': 'train.txt',
        'criteo_test': 'test.txt',
        # Movielens - 1M
        'ml_base': 'data/movielens',
        'ml_train': 'ratings.dat',
        'ml_movie': 'movies.dat',
        'ml_users': 'users.dat',
        # Avazu
        'avazu_base': 'data/avazu',
        'avazu_train': 'train.csv',
        # taobao click
        'tbadclick_base': 'data/tb_adclick',
        'tbadclick_train': 'raw_sample.csv',
        # fliggy click
        'fliggy_base': 'data/fliggy',
        'fliggy_train': 'user_item_behavior_history.csv',
        'fliggy_columns': {
            'query': ['C1]]query', 'I1', 'I2', 'I3', 'I4', 'C3', 'C4]]city', 'S1'],
            'item': ['C2]]item', 'C5', 'C6]]city', 'S2'],
            'query_id': 'C1]]query',
            'item_id': 'C2]]item',
            'target_id': 'S3]]item',
            # 对于部分数据集的 target columns 做 pad_sequence 处理耗费十分巨大，所以是否在训练之前删除对应的列
            'drop_target': True,
        },
        # huawei dataset:
        'huawei_base': 'data/huawei',
        'huawei_train': 'train/train_data_ads.csv',
        # movielens - 100k
        'ml100k_base': 'data/ml100k',
        'ml100k_train': 'u.data',
        'ml100k_columns': {
            'query': ['C1]]query', 'I1', 'I2', 'C2', 'C3', 'S1]]item', 'S2]]item'],
            'item': ['C4]]item', 'C5', 'S3]]genre', 'I3'],
            'query_id': 'C1]]query',
            'item_id': 'C4]]item',
            'target_id': 'S1]]item',  # topk 对比的列表
        },
    }
}
