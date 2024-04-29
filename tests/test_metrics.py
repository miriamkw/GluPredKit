import pytest
from glupredkit.metrics.base_metric import BaseMetric
from glupredkit.metrics.rmse import Metric as RMSE
from glupredkit.metrics.mae import Metric as MAE
from glupredkit.metrics.pcc import Metric as PCC


# Fixture for shared input and target data
@pytest.fixture
def data():
    input_data = [1, 2, 3, 4, 5]
    target_data = [1, 3, 5, 7, 9]
    return input_data, target_data


# Test cases for all metrics
@pytest.mark.parametrize("metric_cls, expected_output", [
    (RMSE, 2.449489742783178),
    (MAE, 2.0),
    (PCC, 1.0)
])
def test_metrics(metric_cls, expected_output, data):
    input_data, target_data = data
    metric = metric_cls()
    output = metric(input_data, target_data)
    assert abs(output - expected_output) < 0.0001, f"Failed for {metric_cls.__name__}"


def test_base_metric_instantiation():
    # Test that BaseMetric cannot be instantiated
    with pytest.raises(TypeError):
        metric = BaseMetric("")


# TODO: What happens if input values are negative, nan, 0 ...?
# TODO: What happens if input values are of not equal length?
# TODO: What happens if input values are strings?
