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

* CTR models
    1. [DeepFM 2017](https://arxiv.org/pdf/1703.04247.pdf)
    2. [DCN 2017](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754?ref=https://githubhelp.com)
    3. [DCN-v2 2020](https://arxiv.org/pdf/2008.13535.pdf)
    4. [AutoInt 2018](https://arxiv.org/pdf/1810.11921.pdf)
    5. [InterHAt 2020](https://dl.acm.org/doi/pdf/10.1145/3336191.3371785)
    6. [GateNet 2020](https://arxiv.org/pdf/2007.03519.pdf)
    7. [EDCN 2021](https://dl.acm.org/doi/pdf/10.1145/3459637.3481915)
    8. [MaskNet 2021](https://arxiv.org/pdf/2102.07619)

* Recall models
  1. [YoutubeDNN](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

* Other:
    1. [CowClip 2022](https://arxiv.org/pdf/2204.06240)

## Dataset

1. criteo
2. avazu
3. [movielens-1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip)
4. [User Behavior Data from Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
5. [Ad Display/Click Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56#1)
6. [Fliggy](https://tianchi.aliyun.com/dataset/dataDetail?dataId=113649)
7. [Huawei Competition](https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html) --- [data_3.zip](https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/2022/AI/238143c8f2b211ec84305c80b6c7dac5/2022_3_data.zip)

## Code Reference

1. [DeepCTR](https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr)
2. [Cowclip](https://github.com/bytedance/LargeBatchCTR)
3. [FuxiCTR](https://github.com/xue-pai/FuxiCTR)
4. [DeepMatch](https://github.com/shenweichen/deepmatch)

## Appendix

### Tips

1. Without MirroredStrategy in lower tf version, it will come with unknown shape warning, even sometimes it will affect
   the training speed. You can ignore the warning.
2. In the Callback(keras.callbacks.Callback) of lower version of tensorflow (<2.4.0), if
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
8. For recommendation system, the numerical feature have large variance and hurt algorithms, so we should normalize
   them(AutoInt C5.1). If we consider training numerical feature with sparse feature in the attention based model, the
   performance may not well.
9. In EDCN, using `BatchNormalization` is required, other than it will occur **gradient vanishing** problems because small
   number matrix hadamard product in bridge module, and use **residual shortcut** optionally.

### Train method

1. Training with 10240, 4096, 1024, 10240 batches in order which can reach better performance.