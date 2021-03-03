import os
import joblib

import cough_quality_check.cough_quality_log as quality_log
import cough_quality_check.CoughConfig as CghCfg

logger = quality_log.get_cough_logger('CoughIntegrityModelLoaderLogger')


class CoughModelLoader(object):
    def __init__(self):
        self.cough_model = None

    def load_model(self):

        logger.info('Loading integrity model')
        try:
            self.cough_model = joblib.load(os.path.join(CghCfg.COUGH_FOLDER, 'CoughModelLoader', CghCfg.COUGH_MODEL))
        except Exception as e:
            logger.error(f'Integrity model loading failed with:\n{e}')
