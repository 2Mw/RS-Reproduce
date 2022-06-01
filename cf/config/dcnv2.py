config = {
    'files': {
        'criteo_base': 'data/criteo',
        'criteo_train': 'train.txt',
        'criteo_test': 'test.txt',
    },

    'train': {
        'epochs': 10,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 64,
        'lr': 0.0005,
        'lr_embed': 1e-4,
        'warmup': True,
        'test_ratio': 1 / 7,
        'test_batch_size': 9012
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 8,
        'l2_reg_embedding': 1e-5,
        'numeric_same_dim': False,  # 表示和categorical数据维度一致
        'embedding_device': 'gpu',
        # deep part
        'hidden_units': [400, 400, 400],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
