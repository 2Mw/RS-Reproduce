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
        'sample_size': 100000,
        'batch_size': 4096,
        'lr': 0.0005,
        'val_ratio': 1 / 14,
        'test_ratio': 1 / 14,
        'test_batch_size': 9012,
        # cowclip params
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': True,
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        'numeric_same_dim': False,
        # Linear residual:
        'linear_res': False,
        # deep part
        'hidden_units': [400, 400, 400],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
