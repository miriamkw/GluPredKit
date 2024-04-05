from .base_metric import BaseMetric
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MARE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mare = np.nanmean(np.abs((y_true - y_pred) / y_true))
        mare = mare * 100

        return mare
