from .base_metric import BaseMetric
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MRE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mre = np.nanmean((y_pred - y_true) / y_true)

        return mre
