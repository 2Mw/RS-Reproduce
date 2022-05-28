# Recommender System

The implementation of various recommender system models (Tensorflow).

## 1. The architecture of documents

```
RS
├─cf                    # collaborative filtering based recommender system
│  ├─config             # store the configurations of models (one model may has many configurations)
│  ├─layers             # store the layers source code
│  ├─models             # store the models source code
│  ├─preprocess         # store the preprocessing of datasets.
│  ├─result             # store the result of training models, include configs, model summays, weights and accuracy (1 -> n)
│  ├─run                # store the files which run the training process.
│  ├─tune               # The multiple config of yaml files which used to fine tune.
│  └─utils              # store some common functions
├─data
│  ├─avazu
│  ├─criteo
│  └─movielens
└─gnn                   # gnn based recommender system
└─log                   # store the log file.
```

## 2. The list of reproducing models

* CF-based
    1. [DeepFM 2017](https://arxiv.org/pdf/1703.04247.pdf)
    2. [DCN 2017](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754?ref=https://githubhelp.com)
    3. [DCN-v2 2020](https://arxiv.org/pdf/2008.13535.pdf)
    4. [AutoInt 2018](https://arxiv.org/pdf/1810.11921.pdf)
    5. [InterHAt 2020](https://dl.acm.org/doi/pdf/10.1145/3336191.3371785)