import numpy as np
from .base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self, name):
        super().__init__(name)
        self.name = "PCC"

    def _calculate_metric(self, y_true, y_pred):
        corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
        return corr_coef
