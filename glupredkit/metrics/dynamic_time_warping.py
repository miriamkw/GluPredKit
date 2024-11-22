import numpy as np
from .base_metric import BaseMetric
from fastdtw import fastdtw


class Metric(BaseMetric):
    def __init__(self):
        super().__init__("DTW")

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        distance, _ = fastdtw(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
        return distance

