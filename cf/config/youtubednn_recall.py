from cf.config.dataset import ds_config

config = {
    'train': {
        'batch_size': 1024
    },

    'model': {
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
