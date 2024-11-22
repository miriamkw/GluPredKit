import numpy as np

from .base_metric import BaseMetric
from error_grids import zone_accuracy
from methcomp import parkeszones


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Parkes Error Grid Linear')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        scores = parkeszones(1, y_true, y_pred, units="mgdl", numeric=True)
        return np.mean(scores)

