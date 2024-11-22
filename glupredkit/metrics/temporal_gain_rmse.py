from .base_metric import BaseMetric
import numpy as np
from scipy.signal import correlate


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Temporal Gain')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        prediction_horizon = kwargs.get('prediction_horizon', 120)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Define the range of lags (in samples) based on the prediction horizon
        lag_range = range(-prediction_horizon // 5, prediction_horizon // 5 + 1)  # Assuming 5s sampling rate

        min_rmse = float('inf')
        best_lag = None

        for lag in lag_range:
            # Shift y_pred by the current lag
            if lag < 0:
                shifted_pred = y_pred[-lag:]  # truncate leading part
                shifted_true = y_true[:len(shifted_pred)]
            else:
                shifted_true = y_true[lag:]  # truncate leading part
                shifted_pred = y_pred[:len(shifted_true)]

            # Compute RMSE for the current lag
            rmse = np.sqrt(np.mean((shifted_true - shifted_pred) ** 2))

            if rmse < min_rmse:
                min_rmse = rmse
                best_lag = lag

        # Convert lag to time units and return
        best_lag_minutes = prediction_horizon + best_lag * 5  # Assuming 5 min sampling rate

        return best_lag_minutes
