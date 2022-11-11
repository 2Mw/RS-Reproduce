import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding, form_x, model_summary
from tensorflow import keras
from cf.layers.mask import MaskedEmbeddingsAggregator as MEA
from cf.layers.ytb import L2Norm
import os


# YouTube DNN 分为两部分：recall 和 rank， this is recall part.

class YoutubeDNNRecall(Model):
    """
    WARNING: This model must train in eager mode.
    Single tower for listwise.
    """
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_columns = feature_columns
        model_cfg = config['model']
        train_cfg = config['train']
        self.directory = directory
        self.embedding_dim = model_cfg['embedding_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim, mask_zero=True)
        self.mask_agg = MEA(name='aggregate_embedding')
        self.mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])
        self.l2 = L2Norm()
        self.bn = BatchNormalization()
        self.pred_class_num = self.__get_pred_class__()
        self.final = Dense(self.pred_class_num, activation=tf.nn.softmax, name='dense_output')

    def __get_pred_class__(self) -> int:
        """
        召回训练数据集分为三部分, user, item, context，是对于 item 类别的多分类任务，需要找到 item 的 vocab_size
        :param: 预测类别的字段名称，只要是属于预测类别的属性就可以
        :return: the number of item class
        """
        for i in self.feature_columns:
            if '::' in i.name and i.name.split('::')[1] == 'item':
                return i.vocab_size
        raise ValueError('Not found the item feature alias')

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        inputs = {
            f.name: Input(shape=(None, ), name=f.name)
            for f in self.feature_columns
        }
        model = Model(inputs, outputs=self.call(inputs))
        if len(self.directory) > 0:
            keras.utils.plot_model(model, os.path.join(self.directory, 'model.png'), show_shapes=True)
        model.summary()

    def call(self, inputs, training=None, mask=None):
        ebd_x = []
        dense_x = []
        seq_x = []
        for f, v in inputs.items():
            key = f
            if '::' in f:  # Get the key
                key = f.split('::')[1]
            if f[0] == 'C':
                # 处理稀疏型特征
                # ebd_x.append(self.ebd[key](v))
                pass
            elif f[0] == 'I':
                # 处理数值型特征
                # v = tf.expand_dims(v, 1)
                # dense_x.append(v)
                pass
            elif f[0] == 'S':
                seq_x.append(self.mask_agg(self.l2(self.ebd[key](v))))

        x = tf.concat(ebd_x + dense_x + seq_x, axis=1)
        x = self.mlp(x)
        return self.final(self.bn(x))


