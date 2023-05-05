import unittest
from test_metric import TestMetric

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add tests to the suite
    test_suite.addTest(unittest.makeSuite(TestMetric))

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(test_suite)