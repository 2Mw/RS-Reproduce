config = {
    "files": {
        "criteo_base": "data/criteo",
        "criteo_train": "train.txt",
        "criteo_test": "test.txt",
    },

    "train": {
        "epochs": 3,
        "optimizer": 'Adam',
        'loss': 'binary_crossentropy',
        "sample_size": 15000000,
        "batch_size": 512,
        "lr": 0.001,
        'val_ratio': 0.083,
        'test_ratio': 0.07,
        'test_batch_size': 9012
    },

    "model": {
        "hidden_units": [256, 128, 64],
        "dropout": 0.5,
        "use_bn": False,
        "fm_w_reg": 0.,
        "embedding_reg": 0.,
        "activation": "relu",
        "embedding_dim": 8,
        "metrics": ['AUC']
    }
}
