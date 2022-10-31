from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'vocab_size', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dim', 'dtype'])
# SequenceFeat 单个字段含有多值属性
SequenceFeat = namedtuple('SequenceFeat', ['name', 'vocab_size', 'dtype'])
