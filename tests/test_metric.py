import sys
sys.path.insert(0, './src')

import unittest
from metrics.base_metric import BaseMetric
from metrics.rmse import RMSE
from metrics.mae import MAE

class TestMetric(unittest.TestCase):

    def test_base_metric(self):
        # Test that BaseMetric cannot be instantiated
        with self.assertRaises(TypeError):
            metric = BaseMetric()

    def test_rmse_metric(self):
        # Test RMSE using a known input and output
        metric = RMSE()
        input_data = [1, 2, 3, 4, 5]
        target_data = [1, 3, 5, 7, 9]
        expected_output = 2.449489742783178
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output)

    def test_mae_metric(self):
        # Test MAE using a known input and output
        metric = MAE()
        input_data = [1, 2, 3, 4, 5]
        target_data = [1, 3, 5, 7, 9]
        expected_output = 2.0
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()