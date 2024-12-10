from .base_metric import BaseMetric
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MRE')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = (y_pred - y_true) / y_true
            # Replace invalid values (resulting from division by zero) with NaN
            relative_error[np.isinf(relative_error) | np.isnan(relative_error)] = np.nan

        mre = np.nanmean(relative_error)

        return mre
