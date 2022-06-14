import cf.models
import cf.run as obj
from cf.run.run_autoint import *
from cf.run.run_can import *
from cf.run.run_dcn import *
from cf.run.run_dcnv2 import *
from cf.run.run_deepfm import *
from cf.run.run_interhat import *
from cf.run.run_edcn import *
from cf.run.run_medcn import *


class Instance:
    def __init__(self, name):
        module = getattr(obj, f'run_{name}')
        self.train = getattr(module, 'train')
        self.evaluate = getattr(module, 'evaluate')


MODULES = {k: Instance(k) for k in cf.models.MODULES.keys()}
