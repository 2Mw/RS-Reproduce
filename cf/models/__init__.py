from cf.models.dcn import *
from cf.models.dcnv2 import *
from cf.models.experimental.can import *
from cf.models.deepfm import *
from cf.models.interhat import *
from cf.models.autoint import *
from cf.models.edcn import *
from cf.models.experimental.medcn import *
from cf.models.experimental.dcn_me import *
from cf.models.experimental.autoint_me import *

MODULES = {
    'dcn': DCN,
    'can': CAN,
    'dcnv2': DCNv2,
    'deepfm': DeepFM,
    'interhat': InterHAt,
    'autoint': AutoInt,
    'edcn': EDCN,
    'medcn': MEDCN,
    'dcn_me': DCNME,
    'autoint_me': AutoIntME
}
