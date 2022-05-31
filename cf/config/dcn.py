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
        "sample_size": -1,
        "batch_size": 4096,
        "lr": 0.001,
        'val_ratio': 0.076,
        'test_ratio': 0.07,
        'test_batch_size': 9012
    },

    "model": {
        # embeddings
        "embedding_reg": 0.,
        "embedding_dim": 8,
        'numeric_same_dim': False,  # 表示和categorical数据维度一致
        'embedding_device': 'gpu',
        # deep part
        "hidden_units": [512, 256, 128, 64],
        "dropout": 0.5,
        "fm_w_reg": 0.,
        "activation": "relu",
        # cross part
        'cross_layer': 3,
        "cross_w_reg": 0.,
        "cross_b_reg": 0.,
        # evaluate part
        "metrics": ['AUC']
    }
}
