import sys
import logging


stream_handler = logging.StreamHandler()
stream_handler.setStream(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

api_logger = logging.getLogger('CoughCheckLogger')
api_logger.setLevel('DEBUG')
api_logger.addHandler(stream_handler)

api_logger = logging.getLogger('CoughIntegrityModelLoaderLogger')
api_logger.setLevel('DEBUG')
api_logger.addHandler(stream_handler)


def get_cough_logger(name):
    return logging.getLogger(name)
