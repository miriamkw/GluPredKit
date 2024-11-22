import numpy as np
from .base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self):
        super().__init__("SNR")

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        signal_power = np.sum((y_true - np.mean(y_true)) ** 2)
        noise_power = np.sum((y_true - y_pred) ** 2)
        if noise_power == 0:
            return np.inf  # Infinite SNR if no noise
        return signal_power / noise_power


