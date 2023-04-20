import unittest
#from my_metrics_module import BaseMetric, RMSEMetric, MAEMetric

class TestMetric(unittest.TestCase):
    """
    def test_base_metric(self):
        # Test that BaseMetric cannot be instantiated
        with self.assertRaises(TypeError):
            metric = BaseMetric()

    def test_rmse_metric(self):
        # Test RMSEMetric using a known input and output
        metric = RMSEMetric()
        input_data = [1, 2, 3, 4, 5]
        target_data = [1, 3, 5, 7, 9]
        expected_output = 2.23606797749979
        output = metric.evaluate(input_data, target_data)
        self.assertAlmostEqual(output, expected_output)

    def test_mae_metric(self):
        # Test MAEMetric using a known input and output
        metric = MAEMetric()
        input_data = [1, 2, 3, 4, 5]
        target_data = [1, 3, 5, 7, 9]
        expected_output = 2.4
        output = metric.evaluate(input_data, target_data)
        self.assertAlmostEqual(output, expected_output)
    """

if __name__ == '__main__':
    unittest.main()