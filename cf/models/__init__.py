from cf.models.ctr.dcn import *
from cf.models.ctr.dcnv2 import *
from cf.models.ctr.experimental.can import *
from cf.models.ctr.deepfm import *
from cf.models.ctr.interhat import *
from cf.models.ctr.autoint import *
from cf.models.ctr.edcn import *
from cf.models.ctr.experimental.medcn import *
from cf.models.ctr.experimental.dcn_me import *
from cf.models.ctr.experimental.autoint_me import *
from cf.models.recall.YoutubeDNN import *
from cf.models.recall.YoutubeSBC import *

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
    'autoint_me': AutoIntME,
    'youtubednn_recall': YoutubeDNNRecall,
    'youtubednn_rank': None,
    'youtubesbc': YoutubeSBC
}
