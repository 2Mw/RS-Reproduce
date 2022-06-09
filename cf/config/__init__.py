from cf.config.autoint import *
from cf.config.can import *
from cf.config.dcn import *
from cf.config.dcnv2 import *
from cf.config.interhat import *
from cf.config.deepfm import *
from cf.config.edcn import *
from cf.models import MODULES as pool
import cf.config as obj

MODULES = {k: getattr(obj, k).config for k in pool.keys()}

if __name__ == '__main__':
    print(MODULES)
