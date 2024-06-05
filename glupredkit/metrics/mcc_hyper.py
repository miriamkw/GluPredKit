"""
Matthews Correlation Coefficient (MCC).
"""
from .base_metric import BaseMetric
from sklearn.metrics import matthews_corrcoef


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MCC Hyperglycemia Detection')

    def _calculate_metric(self, y_true, y_pred):
        hyper_threshold = 180

        y_true = [val > hyper_threshold for val in y_true]
        y_pred = [val > hyper_threshold for val in y_pred]

        mcc = matthews_corrcoef(y_true, y_pred)

        return mcc
