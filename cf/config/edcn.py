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
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': False,
        # Metrics
        'metrics': ['AUC', 'BCE']
    },
    # anonymecitoyen

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        'numeric_same_dim': False,
        'numeric_process': 'mms',
        'use_embed_gate': False,
        # Linear residual:
        'linear_res': False,
        # deep part
        'hidden_units': [400, 400, 400],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'gelu',
        'use_bn': True,
        'use_residual': False,
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # bridge and regularization
        'bridge_type': 'hadamard_product',
        'use_regulation_module': False,
        'tau': 1,
    }
}

config.update(ds_config)
