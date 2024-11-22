from .base_metric import BaseMetric
import numpy as np
from scipy.signal import correlate


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Temporal Gain')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        # 120 as default
        prediction_horizon = kwargs.get('prediction_horizon', 120)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cross_corr = correlate(y_true, y_pred, mode='full')

        # Define the range of lags that corresponds to the prediction horizon
        lag_min = -prediction_horizon // 5
        lag_max = prediction_horizon // 5

        # Calculate the full range of lags for the cross-correlation
        full_lags = np.arange(-len(y_true) + 1, len(y_pred))

        # Filter the valid lags within the prediction horizon
        valid_mask = (full_lags >= lag_min) & (full_lags <= lag_max)
        valid_lags = full_lags[valid_mask]
        valid_cross_corr = cross_corr[valid_mask]

        # Find the lag with the maximum cross-correlation within the prediction horizon
        max_corr_idx = np.argmax(np.abs(valid_cross_corr))
        lag = prediction_horizon + valid_lags[max_corr_idx] * 5

        return lag

