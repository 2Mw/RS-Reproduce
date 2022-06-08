from cf.config.dataset import ds_config

config = {
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
        # Metrics
        'metrics': ['AUC', 'BCE']
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        'numeric_same_dim': True,
        'numeric_process': 'mms',
        'use_embed_gate': False,
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
        'low_rank': 256,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # attention layer
        'att_layer_num': 3,
        'att_head_num': 2,
        'att_size': 8,
        'att_dropout': 0.5,
    }
}

config.update(ds_config)
