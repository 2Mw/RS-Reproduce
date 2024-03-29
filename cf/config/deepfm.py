from cf.config.dataset import ds_config

config = {
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
        'cowclip': False,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': True,
        # Metrics
        'metrics': ['AUC', 'BCE']
    },

    'model': {
        # Embedding params
        'numeric_same_dim': False,
        'numeric_process': 'mms',
        'use_embed_gate': False,
        'embedding_device': 'gpu',
        'embedding_reg': 0.,
        'embedding_dim': 10,
        # Linear residual:
        'linear_res': False,
        # DNN params
        'hidden_units': [256, 128, 64],
        'dropout': 0.5,
        'use_bn': False,
        'fm_w_reg': 0.,
        'activation': 'relu',
    }
}

config.update(ds_config)
