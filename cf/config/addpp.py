from cf.config.dataset import ds_config

__experts__ = 4

config = {
    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'sparse_categorical_crossentropy',
        'sample_size': -1,
        'batch_size': 64,
        'lr': 0.01,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'test_batch_size': 2048,
        'lr_embed': 1e-4,
        'warmup': True,
        'metrics': []
    },

    'model': {
        'numeric_process': 'mms',
        # temperature factor for similarity score, default 1.0.
        'temperature': 0.5,
        'activation': 'PReLU',
        'dropout': 0.3,
        'use_bn': False,
        'units': [256, 256, 256],
        # number of interests
        # MMoE config: 1. embedding mmoe 2. dnn mmoe
        'mmoe_experts': __experts__,
        'use_dnn_mmoe': True,
        'bnn_bridge_type': 'concat',
        'merge_strategy': 'dense',   # mean or dnn

    }
}

config.update(ds_config)