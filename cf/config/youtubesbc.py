from cf.config.dataset import ds_config

config = {
    'train': {
        'batch_size': 1024
    },

    'model': {
        # the number of negative sample for every positive sample, default to 3.
        # Note that it must smaller than batch_size
        'n_neg': 3,
        # temperature factor for similarity score, default 1.0.
        'temperature': 1.0,
        # the params for user and item tower
        'user_params': {
            'dims': [256, 128, 64],
            'activation': 'prelu'
        },
        'item_params': {
            'dims': [256, 128, 64],
            'activation': 'prelu'
        }
    }
}

config.update(ds_config)
