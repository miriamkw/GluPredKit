"""
Matthews Correlation Coefficient (MCC).
"""
from .base_metric import BaseMetric
from sklearn.metrics import recall_score
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('G-Mean')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        hypo_threshold = 70
        hyper_threshold = 180

        true_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_true]
        predicted_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_pred]

        recalls = recall_score(true_labels, predicted_labels, average=None)

        # Replace zeros with a very small number instead of raising error
        epsilon = 1e-10  # Small constant
        recalls = np.where(recalls == 0, epsilon, recalls)

        # Calculate geometric mean
        g_mean_value = np.exp(np.mean(np.log(recalls)))
        return g_mean_value

