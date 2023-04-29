import unittest
from src.metrics.base_metric import BaseMetric
from src.metrics.rmse import RMSE
from src.metrics.mae import MAE
from src.metrics.van_herpe import VanHerpe
from src.metrics.kovatchev import Kovatchev
from src.metrics.cao import Cao
from src.metrics.bayer import Bayer
from src.metrics.pcc import PCC

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

    def test_pcc_metric(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 5, 7, 9]
        expected_output = 1.0
        metric = PCC()
        output = metric(y_true, y_pred)
        self.assertAlmostEqual(output, expected_output)


    def test_mae_metric(self):
        # Test MAE using a known input and output
        metric = MAE()
        input_data = [1, 2, 3, 4, 5]
        target_data = [1, 3, 5, 7, 9]
        expected_output = 2.0
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output)

    def test_van_herpe_metric(self):
        metric = VanHerpe()
        input_data = [90]
        target_data = [140]
        expected_output = 34.675294
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output, delta=0.0001)

    def test_kovatchev_metric(self):
        metric = Kovatchev()
        input_data = [90]
        target_data = [140]
        expected_output =  10.169543
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output, delta=0.0001)
    
    def test_cao_metric(self):
        metric = Cao()
        input_data = [90]
        target_data = [140]
        expected_output = 40.02489
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output, delta=0.0001)

    def test_bayer_metric(self):
        metric = Bayer()
        input_data = [90]
        target_data = [140]
        expected_output = 7.71737
        output = metric(input_data, target_data)
        self.assertAlmostEqual(output, expected_output, delta=0.0001)

if __name__ == '__main__':
    unittest.main()