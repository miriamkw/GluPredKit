import pytest
import numpy as np
from glupredkit.metrics.base_metric import BaseMetric
from glupredkit.metrics.clarke_error_grid import Metric as ClarkeErrorGrid
from glupredkit.metrics.glycemia_detection import Metric as GlycemiaDetection
from glupredkit.metrics.grmse import Metric as gRMSE
from glupredkit.metrics.mae import Metric as MAE
from glupredkit.metrics.mare import Metric as MARE
from glupredkit.metrics.mcc_hyper import Metric as MCCHyper
from glupredkit.metrics.mcc_hypo import Metric as MCCHypo
from glupredkit.metrics.me import Metric as ME
from glupredkit.metrics.mre import Metric as MRE
from glupredkit.metrics.parkes_error_grid import Metric as ParkesErrorGrid
from glupredkit.metrics.parkes_error_grid_exp import Metric as ParkesErrorGridExp
from glupredkit.metrics.pcc import Metric as PCC
from glupredkit.metrics.rmse import Metric as RMSE
from glupredkit.helpers.unit_config_manager import unit_config_manager

# Defining the list of metric classes
metric_classes = [
    ClarkeErrorGrid,
    GlycemiaDetection,
    gRMSE,
    MAE,
    MARE,
    MCCHyper,
    MCCHypo,
    ME,
    MRE,
    ParkesErrorGrid,
    ParkesErrorGridExp,
    PCC,
    RMSE
]


@pytest.fixture
def data():
    input_data = [100, 105, 110, 115, 120]
    target_data = [98, 107, 109, 120, 119]
    return input_data, target_data


# Test cases for all metrics
@pytest.mark.parametrize("metric_cls, expected_output", [
    (ClarkeErrorGrid, ['100.0%', '0.0%', '0.0%', '0.0%', '0.0%']),
    (GlycemiaDetection, [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
    (gRMSE, 2.6457513110645907),
    (MAE, 2.2),
    (MARE, 1.9668818397632564),
    (MCCHyper, 0.0),
    (MCCHypo, 0.0),
    (ME, -0.6),
    (MRE, -0.004474483783052791),
    (ParkesErrorGrid, ['100.0%', '0.0%', '0.0%', '0.0%', '0.0%']),
    (ParkesErrorGridExp, 0.9999),
    (PCC, 0.952818526928451),
    (RMSE, 2.6457513110645907)
])
def test_metrics(metric_cls, expected_output, data):
    # Note which unit is used in settings
    use_mgdl = unit_config_manager.use_mgdl

    # Use mg/dL for this test
    unit_config_manager.use_mgdl = True

    input_data, target_data = data
    metric = metric_cls()
    output = metric(target_data, input_data)

    # Set the setting back to the user defined setting
    unit_config_manager.use_mgdl = use_mgdl

    assert output == expected_output, f"Metric value test failed for {get_metric_name(metric_cls)}."


def test_base_metric_instantiation():
    # Test that BaseMetric cannot be instantiated
    with pytest.raises(TypeError):
        _ = BaseMetric("")


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_metric_class_name(metric_cls):
    metric = metric_cls()
    assert metric.__class__.__name__ == "Metric", f"Class name for {metric_cls.__name__} is not 'Metric'"


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_negative_values(metric_cls):
    input_data_negative = [-100, -105, -110, -115, -120]
    target_data_negative = [-98, -107, -109, -120, -119]
    target_data_positive = [98, 107, 109, 120, 119]

    metric = metric_cls()

    # Test negative target data
    with pytest.raises(ValueError):
        metric(target_data_negative, input_data_negative)

    # Test negative values to ensure that the metric still produces an output
    output_negative = metric(target_data_positive, input_data_negative)

    assert not check_for_nan(output_negative), (f"The metric {get_metric_name(metric_cls)} fails to produce a value "
                                                f"with negative inputs.")


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_nan_values(metric_cls):
    input_data_nan = [100, np.nan, 110, np.nan, 120]
    target_data_nan = [98, 107, np.nan, 120, np.nan]

    metric = metric_cls()

    # Ensure metric calculation with NaN values issues a warning but does not raise an exception
    with pytest.warns(UserWarning, match="NaN values detected"):
        output_nan = metric(target_data_nan, input_data_nan)

    # Ensure the output is not NaN
    assert not check_for_nan(output_nan), (
        f"The metric {get_metric_name(metric_cls)} fails to produce a value with NaN "
        f"values in input")


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_zero_values(metric_cls):
    input_data_zero = [0, 0, 0, 0, 0]
    target_data_zero = [0, 0, 0, 0, 0]
    target_data_positive = [98, 107, 109, 120, 119]

    metric = metric_cls()

    # Test zero in target data
    with pytest.raises(ValueError):
        metric(target_data_zero, input_data_zero)

    # Test predicted zero values to ensure that the metric still produces an output
    output_negative = metric(target_data_positive, input_data_zero)
    assert not check_for_nan(output_negative), (f"The metric {get_metric_name(metric_cls)} fails to produce a value "
                                                f"with zero in inputs.")


def check_for_nan(value):
    if isinstance(value, float):
        return np.isnan(value)
    elif isinstance(value, list) or isinstance(value, tuple):
        # Pass the test if outputs is of list format
        return False
    elif isinstance(value, np.ndarray):
        # Check if any NaN value exists in the array
        return np.isnan(value).any()
    else:
        raise TypeError(f"Unhandled type for NaN check: {type(value)}")


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_unequal_length(metric_cls):
    input_data_unequal_length = [100, 105, 110, 115]
    target_data_unequal_length = [98, 107, 109, 120, 119]

    metric = metric_cls()

    # Test unequal length values
    with pytest.raises(ValueError):
        metric(target_data_unequal_length, input_data_unequal_length)


@pytest.mark.parametrize("metric_cls", metric_classes)
def test_string_values(metric_cls):
    input_data_strings = ["100", "105", "110", "115", "120"]
    target_data_strings = ["98", "107", "109", "120", "119"]

    metric = metric_cls()

    # Test string values
    with pytest.raises(TypeError):
        metric(target_data_strings, input_data_strings)


def get_metric_name(metric_class):
    return metric_class.__module__.split('.')[-1]
