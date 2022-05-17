config = {
    "files": {
        "criteo_base": "E:/Notes/DeepLearning/dataset/criteo",
        "criteo_train": "train.txt",
        "criteo_test": "test.txt",
    },

    "train": {
        "epochs": 300,
        "optimizer": 'Adam',
        'loss': 'binary_crossentropy',
        "sample_size": 4500000,
        "batch_size": 4096,
        "lr": 0.001,
    },

    "model": {
        # embeddings
        "embedding_reg": 0.,
        "embedding_dim": 8,
        # deep part
        "hidden_units": [256, 128, 64],
        "dropout": 0.5,
        "fm_w_reg": 0.,
        "activation": "relu",
        # cross part
        'cross_layer': 4,
        "cross_w_reg": 0.,
        "cross_b_reg": 0.,
        # evaluate part
        "metrics": ['AUC', 'BCE']
    }
}
