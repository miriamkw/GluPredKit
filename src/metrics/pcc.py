import numpy as np
from .base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self):
        super().__init__("PCC")

    def _calculate_metric(self, y_true, y_pred):
        corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
        return corr_coef
