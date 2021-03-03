import numpy as np

import cough_quality_check.cough_quality_log as cough_log
import cough_quality_check.CoughConfig as CghCfg
from cough_quality_check.CoughModelLoader import CoughModelLoader

main_logger = cough_log.get_cough_logger('CoughCheckLogger')


class CoughCheck(object):
    def __init__(self):
        self.ML = CoughModelLoader()

    @staticmethod
    def get_version():
        version = CghCfg.MODEL_VERSION
        return version

    def integrity_classification(self, embeddings):
        probability = self.ML.cough_model.predict_proba(embeddings)[:, 1][0]
        result = int(probability > CghCfg.INTEGRITY_THRESHOLD)
        return result, np.round(probability, 3)

    def check_audio(self, emb, config=None):

        output_object = {
            'status': {
                'check_passed': False,
                'rejection_reason': None,
                'failure_reason': None,
            },
            'meta':
            {
                'check_module_version': self.get_version()
            },
            'results': {
                'cough_integrity': None
            }
        }

        # default config
        if config is None:
            config = {'use_hops': False}

        # loading cough integrity model
        self.ML.load_model()

        try:
            if config['use_hops']:
                arr_emb = np.array(emb)
                if len(arr_emb) > 1:
                    avg_emb = np.average(arr_emb, axis=0).reshape(1, -1)
                    integrity_result, integrity_proba = self.integrity_classification(avg_emb)
                else:
                    integrity_result, integrity_proba = self.integrity_classification(emb)
            else:
                integrity_result, integrity_proba = self.integrity_classification(emb)
            output_object['results']['cough_integrity'] = integrity_proba
            if integrity_result != 1:
                main_logger.info('Recording not passed the cough check.')
                output_object['status']['rejection_reason'] = 'COUGH_CHECK_NOT_PASSED'
        except Exception as e:
            output_object['status']['failure_reason'] = 'COUGH_PROCESSING_FAILED'
            main_logger.warning(f'Cough validation failed with:\n{e}')

        main_logger.info(f'Cough check results: {output_object}')
        return output_object
