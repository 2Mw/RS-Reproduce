config = {
    "files": {
        "criteo_base": "data/criteo",
        "criteo_train": "train.txt",
        "criteo_test": "test.txt",
    },

    "train": {
        "epochs": 2,
        "optimizer": 'Adam',
        'loss': 'binary_crossentropy',
        "sample_size": 10000000,
        "batch_size": 4096,
        "lr": 0.001,
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
