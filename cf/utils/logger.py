import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', datefmt='%y%m%d%H%M%S')
logger = logging.getLogger('RS')
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.getLevelName('debug'))

if __name__ == '__main__':
    logger.info('asd')
