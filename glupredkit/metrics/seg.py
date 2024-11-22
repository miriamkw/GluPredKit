import numpy as np

from .base_metric import BaseMetric
from methcomp import segscores


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Surveillance Error Grid Score')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        y_true = [1 if val < 1 else 599 if val > 599 else val for val in y_true]
        y_pred = [1 if val < 1 else 599 if val > 599 else val for val in y_pred]
        scores = segscores(y_true, y_pred, units="mgdl")
        return np.mean(scores)

