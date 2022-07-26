from cf.config.dataset import ds_config

config = {
    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 4096,
        'lr': 0.001,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
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
        # preprocess
        'numeric_process': 'mms',
        # embeddings
        'embedding_dim': 10,
        'embedding_device': 'gpu',
        'embedding_reg': 0.,
        'l2_reg_embedding': 1e-5,
        'numeric_same_dim': False,
        # Linear residual:
        'linear_res': False,
        # deep part
        'hidden_units': [400, 400, 400],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': True,
        'use_residual': False,
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # bridge and broker
        'bridge_type': 'concat',
        'broker_experts': 4,
        'broker_gates': 2,
        'using_embedding_broker': True,
        'using_feature_broker': True,
        # sequence feature
        'seq_split': '^',
    }
}

config.update(ds_config)
