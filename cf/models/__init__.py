import sys

import cf
from cf.models.dcn import *
from cf.models.dcnv2 import *
from cf.models.can import *
from cf.models.deepfm import *
from cf.models.interhat import *
from cf.models.autoint import *

MODULES = {
    'dcn': DCN,
    'can': CAN,
    'dcnv2': DCNv2,
    'deepfm': DeepFM,
    'interhat': InterHAt,
    'autoint': AutoInt
}
