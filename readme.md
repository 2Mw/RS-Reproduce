# Recommender System

The implementation of various recommender system models (Tensorflow).

## 1. The architecture of the project

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

* Other:
    1. [CowClip](https://arxiv.org/pdf/2204.06240)

## Dataset

1. criteo
2. [movielens-1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip)
3. [User Behavior Data from Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
4. [Ad Display/Click Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56#1)
5. [Fliggy](https://tianchi.aliyun.com/dataset/dataDetail?dataId=113649)

## Reference

1. [DeepCTR](https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr)
2. [Cowclip](https://github.com/bytedance/LargeBatchCTR)

## Appendix

1. Without MirroredStrategy in lower tf version, it will come with unknown shape warning, even sometimes it will affect
   the training speed.
2. In the Callback(keras.callbacks.Callback) of lower version of tensorflow (<=2.4.0), if
   set `self.model.stop_training = True`, the training process will stop in the end of **epoch**, while higher version
   of tf will stop in the end of this **step**.
3. If you wanna to modify the process of keras function `fit()`, you can inherit the class `keras.model.Model`, then
   override the function `train_step()`(Learnt from cowclip model).
4. Some imports such as `from keras.callbacks import Callback` may not work in lower version of tensorflow 2.X(<2.4.0),
   then you should replace it with `from tensorflow.keras.callbacks import Callback`.
5. If you want to debug training process gracefully, you should set the flag `run_eagerly=True` in the `compile()`
   function when construct the model.
6. Original cowclip source code will crash when gradient because there are mismatch between gradient and trainable
   variables(In the `train_step` function).
7. For cowclip model in lower version of tensorflow 2.X(<2.4.0), you should also implement the function `test_step`
   otherwise it will cause OOM error because of bad conversion of tensor.