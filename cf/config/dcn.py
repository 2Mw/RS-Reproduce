config = {
    'files': {
        # Criteo
        'criteo_base': 'data/criteo',
        'criteo_train': 'train.txt',
        'criteo_test': 'test.txt',
        # Movielens
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
        'fliggy_train': 'user_item_behavior_history.csv'
    },

    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 4096,
        'lr': 0.001,
        'val_ratio': 1 / 14,
        'test_ratio': 1 / 14,
        'test_batch_size': 9012,
        # cowclip params
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': True,
        # Metrics
        'metrics': ['AUC', 'BCE']
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'embedding_device': 'gpu',
        'numeric_same_dim': False,
        'use_embed_gate': False,
        # Linear residual:
        'linear_res': False,
        # deep part
        'hidden_units': [512, 256, 128, 64],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        # cross part
        'cross_layer': 3,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
    }
}
