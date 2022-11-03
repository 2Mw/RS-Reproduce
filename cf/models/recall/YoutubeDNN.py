import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from cf.layers import mlp
from cf.models.ctr.base import get_embedding, form_x, model_summary
from keras.losses import cosine_similarity
from cf.layers.mask import MaskedEmbeddingsAggregator as MEA
from cf.layers.ytb import L2Norm


# YouTube DNN 分为两部分：recall 和 rank， this is recall part.

class YoutubeDNNRecall(Model):
    def __init__(self, feature_columns, config, directory="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_columns = feature_columns
        model_cfg = config['model']
        train_cfg = config['train']
        self.directory = directory
        self.embedding_dim = model_cfg['embedding_dim']
        self.ebd = get_embedding(feature_columns, self.embedding_dim)
        self.mask_agg = MEA(name='aggregate_embedding')
        self.mlp = mlp.MLP(model_cfg['units'], model_cfg['activation'], model_cfg['dropout'], model_cfg['use_bn'])
        self.l2 = L2Norm()
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
        model_summary(self, self.feature_columns, self.directory)

    def call(self, inputs, training=None, mask=None):
        # TODO 输入数据时候的 pad_sequence 怎么搞
        pass
