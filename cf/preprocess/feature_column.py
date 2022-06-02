from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'vocab_size', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dim', 'dtype'])

# class SparseFeat:
#     def __init__(self, name, vocab_size, dtype='int32'):
#         """
#         Initialize sparse feature.
#
#         :param name: str, name of the sparse feature
#         :param vocab_size: int, vocabulary size of the feature
#         :param dtype: str, type of the feature
#         """
#         self.name = name
#         self.vocab_size = vocab_size
#         # if embed_dim == 'auto':
#         #     self.embed_dim = 6 * int(pow(vocab_size, 0.25))
#         # else:
#         #     self.embed_dim = embed_dim
#         self.embed_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=2020)
#         self.dtype = dtype
#
#
# class DenseFeat:
#     def __init__(self, name, dimension, transform_fn, dtype='float32'):
#         """
#         Initialize dense feature.
#
#         :param name: str, name of the dense feature
#         :param dimension: int, dimension of the feature
#         :param transform_fn: function, transform function of the feature
#         :param dtype: str, type of the feature
#         """
#         self.name = name
#         self.dimension = dimension
#         self.transform_fn = transform_fn
#         self.dtype = dtype
