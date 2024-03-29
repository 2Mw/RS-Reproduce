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
        'test_batch_size': 4096,
        # cowclip params
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': False,
        # Metrics
        'metrics': ['AUC', 'BCE']
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        'numeric_process': 'mms',
        'numeric_same_dim': True,
        'use_embed_gate': False,
        # Linear residual:
        'linear_res': False,
        # deep part
        'dropout': 0,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        'use_residual': True,
        # attention layer
        'att_layer_num': 3,
        'att_head_num': 8,
        'att_dk': 30,  # dk in Multi-head Attention.
        'att_dropout': 0.1,
        'att_block': 2,
        'regularization': 0.0002,
        'agg_order': 3,
        'agg_filters': 64,
    }
}

config.update(ds_config)
