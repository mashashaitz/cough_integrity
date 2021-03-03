import os
import sys
sys.path.append(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}')
import json
from unittest import TestCase

import numpy as np

from cough_quality_check.CoughCheck import CoughCheck
import cough_quality_check.CoughConfig as CghCfg


class TestCoughCheck(TestCase):
    def setUp(self):
        self.cough_check = CoughCheck()
        self.cough_check.ML.load_model()
        self.derivation = CghCfg.INTEGRITY_DERIVATION
        self.base_path = CghCfg.COUGH_FOLDER + '/Tests/emb/'

        with open("test_data.json", "r") as test_data:
            self.test_data = json.load(test_data)

    def test_version_in_configs(self):
        self.assertEqual(self.cough_check.get_version(), CghCfg.MODEL_VERSION)

    def test_integrity_classification(self):
        for test in self.test_data:
            emb = np.load(''.join([self.base_path + self.test_data[test]['path']]))
            max_target_proba = self.test_data[test]['proba'] * (1 + self.derivation)
            min_target_proba = self.test_data[test]['proba'] * (1 - self.derivation)
            if self.test_data[test]["proba_type"] == "low":
                self.assertEqual(self.cough_check.check_audio(emb)['status']['rejection_reason'],
                                 'COUGH_CHECK_NOT_PASSED')
            self.assertTrue(min_target_proba <= self.cough_check.integrity_classification(emb)[-1] <= max_target_proba)

    def test_empty_embedding(self):
        emb = np.array([])
        with self.assertRaises(ValueError):
            self.cough_check.integrity_classification(emb)

    def test_failed_check_output(self):
        emb = np.array([])
        self.assertEqual(self.cough_check.check_audio(emb)['status']['failure_reason'], 'COUGH_PROCESSING_FAILED')
