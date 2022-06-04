config = {
    'files': {
        'criteo_base': 'data/criteo',
        'criteo_train': 'train.txt',
        'criteo_test': 'test.txt',
    },

    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 7168,
        'lr': 0.001,
        'test_ratio': 1 / 14,
        'test_batch_size': 9012,
        # cowclip params
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': False,
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'embedding_device': 'gpu',
        # deep part
        'hidden_units': [512, 256, 128, 64],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        # cross part
        'cross_layer': 3,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
