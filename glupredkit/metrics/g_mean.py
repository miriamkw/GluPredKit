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

        # Check for any zero recalls
        if np.any(recalls == 0):
            raise ValueError(
                f"One or more regions have zero recall. Recalls: {recalls}. "
                "The error happens because the model completely missed predicting one or more categories (regions), "
                "resulting in a recall of 0 for those categories. This might be due to imbalanced data, poor model "
                "performance, or unsuitable thresholds."
            )

        return np.sqrt(np.prod(recalls))

