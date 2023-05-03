import numpy as np
from src.metrics.base_metric import BaseMetric

class PCC(BaseMetric):
    def __init__(self):
        self.name = "PCC"

    def _calculate_metric(self, y_true, y_pred):
        corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
        return corr_coef