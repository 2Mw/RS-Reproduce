import logging

logger = logging.getLogger('RS')
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.getLevelName('debug'))

if __name__ == '__main__':
    logger.info('asd')
