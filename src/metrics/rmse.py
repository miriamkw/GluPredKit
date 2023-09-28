from src.metrics.base_metric import BaseMetric
import numpy as np

class RMSE(BaseMetric):
    def __init__(self, use_mgdl=True):
        super().__init__('RMSE')

        if use_mgdl:
            self.k = 1
        else:
            self.k = 18.0182

    def _calculate_metric(self, y_true, y_pred, ):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.sqrt(np.mean(np.square(y_true - y_pred))) / self.k
