from .base_metric import BaseMetric
import numpy as np
from glupredkit.config_manager import config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MAE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = np.mean(np.abs(y_true - y_pred))

        if config_manager.use_mgdl:
            return mae
        else:
            return config_manager.convert_value(mae)