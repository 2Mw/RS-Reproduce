config = {
    'files': {
        'criteo_base': 'data/criteo',
        'criteo_train': 'train.txt',
        'criteo_test': 'test.txt',
    },

    'train': {
        'epochs': 3,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': -1,
        'batch_size': 4096,
        'lr': 0.001,
        'test_ratio': 1 / 7,
        'test_batch_size': 9012
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'sample_size': 10,
        'l2_reg_embedding': 1e-5,
        'embedding_device': 'gpu',
        # deep part
        'dropout': 0,
        'fm_w_reg': 0.,
        'activation': 'relu',
        'use_bn': False,
        'use_residual': True,
        # cross part
        'cross_layers': 4,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.,
        'low_rank': 258,
        'num_experts': 4,
        'l2_reg_cross': 1e-5,
        # attention layer
        'att_layer_num': 3,
        'att_head_num': 2,
        'att_size': 8,
        'att_dropout': 0.5,
        # evaluate part
        'metrics': ['AUC', 'BCE']
    }
}
