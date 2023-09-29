from base_metric import BaseMetric
import numpy as np

class RMSE(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.sqrt(np.mean(np.square(y_true - y_pred)))
