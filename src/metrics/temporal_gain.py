from src.metrics.base_metric import BaseMetric
import numpy as np


class TemporalGain(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def _calculate_metric(self, y_true, y_pred, output_offset):
        """
        y_true -- list of blood glucose measurements
        y_pred -- list of predicted values with y_true as reference values
        output_offset -- the amount of minutes between reference and predicted blood glucose values
        """
        N = len(y_true)
        t = 5 # CGM sample rate
        L = int(output_offset / t)

        delays = np.arange(0, L)
        temporal_gains = np.zeros(L)

        for i in delays:
            #squared_diff_sum = np.sum((y_pred[i:N - L - i] - y_true[:N - L]) ** 2)

            squared_diff_sum = np.sum([(x - y) ** 2 for x, y in zip(y_pred[i:N - L - i], y_true[:N - L])])

            temporal_gain = squared_diff_sum / (N - L)

            temporal_gains[i] = temporal_gain

        print(temporal_gains)

        delay = delays[np.argmin(temporal_gains)]

        return (L - delay) * t

