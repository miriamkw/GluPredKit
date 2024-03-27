from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MAE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = np.nanmean(np.abs(y_true - y_pred))

        if unit_config_manager.use_mgdl:
            return mae
        else:
            return unit_config_manager.convert_value(mae)