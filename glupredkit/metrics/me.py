from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('ME')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        me = np.nanmean(y_pred - y_true)

        if unit_config_manager.use_mgdl:
            return me
        else:
            return unit_config_manager.convert_value(me)
