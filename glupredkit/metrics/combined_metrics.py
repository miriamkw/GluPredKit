from glupredkit.metrics.grmse import Metric as Metric1
from glupredkit.metrics.continous_seg import Metric as Metric2
from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Combined Metrics')

    def __call__(self, y_true, y_pred):
        metric1 = Metric1()
        metric2 = Metric2()

        result1 = metric1(y_true, y_pred)
        result2 = metric2(y_true, y_pred)

        min = 0
        max = 600

        if not unit_config_manager.use_mgdl:
            min = unit_config_manager.convert_value(min)
            max = unit_config_manager.convert_value(max)

        # Normalize to [0, 1] if not already normalized
        result1_normalized = (result1 - min) / (max - min)
        # result2_normalized = (result2 - min) / (max - min)

        # Calculate the weighted average of the two metrics
        combined_result = (result1_normalized * 0.9 + result2 * 0.1) / 2

        return combined_result
