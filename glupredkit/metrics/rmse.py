from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        if unit_config_manager.use_mgdl:
            return rmse
        else:
            return unit_config_manager.convert_value(rmse)
