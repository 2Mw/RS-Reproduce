from collections import namedtuple

# 对于共享 Embedding 的字段应该采用 "xxx::key"的形式
SparseFeat = namedtuple('SparseFeat', ['name', 'vocab_size', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dim', 'dtype'])
# SequenceFeat 单个字段含有多值属性
SequenceFeat = namedtuple('SequenceFeat', ['name', 'vocab_size', 'dtype'])
