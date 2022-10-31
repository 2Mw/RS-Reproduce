from cf.config.dataset import ds_config

config = {
    'train': {
        'batch_size': 1024
    },

    'model': {
        # temperature factor for similarity score, default 1.0.
        'temperature': 1.0,
        'embedding_dim': 20,
        'user_params': {
            'units': [256, 128, 64],
            'activation': 'prelu',
            'dropout': 0.5,
            'use_bn': True,
        },
        'item_params': {
            'units': [256, 128, 64],
            'activation': 'prelu',
            'dropout': 0.5,
            'use_bn': True,
        }
    }
}

config.update(ds_config)
