import numpy as np

from .base_metric import BaseMetric
from error_grids import zone_accuracy


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Parkes Error Grid Exponential Cost Function')

    def _calculate_metric(self, y_true, y_pred):
        accuracy_values = zone_accuracy(y_true, y_pred, 'parkes')
        max_score = 10**4
        score = -1
        for i, val in enumerate(accuracy_values):
            score += 10**(4 - i) * val
        result = score / max_score
        return result
