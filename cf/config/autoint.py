config = {
    'files': {
        "criteo_base": "data/criteo",
        "criteo_train": "train.txt",
        "criteo_test": "test.txt",
    },

    'train': {
        "epochs": 10,
        "optimizer": 'Adam',
        'loss': 'binary_crossentropy',
        'sample_size': 20000,
        "batch_size": 4096,
        "lr": 0.001,
        'val_ratio': 0.076,
        'test_ratio': 0.07,
        'test_batch_size': 9012
    },

    'model': {
        # embeddings
        'embedding_reg': 0.,
        'embedding_dim': 25,
        'l2_reg_embedding': 1e-5,
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
        'metrics': ['AUC']
    }
}
