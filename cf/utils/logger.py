import logging
import sys

logger = logging.getLogger('RS')
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s', '%y%m%d%H%M%S')
_default_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(_default_handler)
# h = logging.StreamHandler(sys.stdout)
# h.setFormatter(logging.Formatter())
# logger.setLevel(logging.getLevelName('debug'))

if __name__ == '__main__':
    logger.info('asd')
