"""
Matthews Correlation Coefficient (MCC).
"""
from .base_metric import BaseMetric
from sklearn.metrics import matthews_corrcoef


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('MCC Hyperglycemia Detection')

    def _calculate_metric(self, y_true, y_pred):
        hypo_threshold = 70

        y_true = [val < hypo_threshold for val in y_true]
        y_pred = [val < hypo_threshold for val in y_pred]

        mcc = matthews_corrcoef(y_true, y_pred)

        return mcc
