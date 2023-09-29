from base_metric import BaseMetric
import numpy as np

class MAE(BaseMetric):
    def __init__(self):
        super().__init__('MAE')

    def _calculate_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.mean(np.abs(y_true - y_pred))
