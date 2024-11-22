"""
Matthews Correlation Coefficient (MCC).
"""
from .base_metric import BaseMetric
from sklearn.metrics import matthews_corrcoef
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Matthews Correlation Coefficient')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        hypo_threshold = 70
        hyper_threshold = 180

        true_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_true]
        predicted_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_pred]

        mcc = matthews_corrcoef(true_labels, predicted_labels)
        return mcc

