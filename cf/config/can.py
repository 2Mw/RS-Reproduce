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
        'tbadclick_train': 'raw_sample.csv'
    },

    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 4096,
        'lr': 0.05,
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
        'numeric_same_dim': True,
        # Linear residual:
        'linear_res': False,
        # deep part
        'dropout': 0,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        'use_residual': True,
        'hidden_units': [256, 128, 64],
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # attention layer
        'att_layer_num': 3,
        'att_head_num': 2,
        'att_size': 8,
        'att_dropout': 0.5,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
