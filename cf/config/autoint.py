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
        'batch_size': 4096,
        'lr': 0.001,
        'test_ratio': 1 / 14,
        'test_batch_size': 9012,
        # cowclip params
        'cowclip': True,
        'lr_embed': 1e-4,
        'clip': 1,
        'bound': 1e-5,
        'warmup': True,
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        # deep part
        'hidden_units': [256, 128, 64],
        'dropout': 0.5,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        # attention layer
        'att_layer_num': 3,
        'att_head_num': 2,
        'att_size': 8,
        'att_dropout': 0,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
