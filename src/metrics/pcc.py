import numpy as np
from metrics.base_metric import BaseMetric

class PCC(BaseMetric):
    def __init__(self):
        self.name = "PCC"

    def __call__(self, y_true, y_pred):
        corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
        return corr_coef