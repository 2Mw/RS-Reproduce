from cf.config.dataset import ds_config

config = {
    'train': {
        'epochs': 500,
        'optimizer': 'Adam',
        'loss': 'sparse_categorical_crossentropy',
        'sample_size': -1,
        'batch_size': 128,
        'lr': 0.0005,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'test_batch_size': 128,
        'lr_embed': 1e-4,
        'warmup': True,
    },

    'model': {
        'numeric_process': 'mms',
        # temperature factor for similarity score, default 1.0.
        'temperature': 1.0,
        'embedding_dim': 64,
        'activation': 'relu',
        'dropout': 0.3,
        'use_bn': True,
        'units': [2048, 1024, 512, 256]
    }
}

config.update(ds_config)
