# Recommender System

The implementation of various recommender system models (Tensorflow).

## 1. The architecture of documents

```
E:.
├─cf                    # collaborative filtering based recommender system
│  ├─config             # store the configurations of models (one model may has many configurations)
│  ├─layers             # store the layers source code
│  ├─models             # store the models source code
│  ├─preprocess         # store the preprocessing of datasets.
│  ├─result             # store the result of training models, include configs, model summays, weights and accuracy (1 -> n)
│  ├─run                # store the files which run the training process.
│  └─utils              # store some common functions
├─data                  # store the datasets.
│  ├─avazu
│  ├─criteo
│  └─movielens
└─gnn                   # gnn based recommender system
```

## 2. The list of reproducing models

* CF-based
    1. [DeepFM 2017](https://arxiv.org/pdf/1703.04247.pdf)
    2. [DCN 2017](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754?ref=https://githubhelp.com)
    3. [DCN-v2 2020](https://arxiv.org/pdf/2008.13535.pdf)
    4. [AutoInt 2018](https://arxiv.org/pdf/1810.11921.pdf)