"""
Matthews Correlation Coefficient (MCC).
"""
from .base_metric import BaseMetric
from sklearn.metrics import f1_score
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Weighted F1 Score')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        hypo_threshold = 70
        hyper_threshold = 180

        true_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_true]
        predicted_labels = [0 if val < hypo_threshold else (2 if val > hyper_threshold else 1) for val in y_pred]

        # Compute Weighted F1-Score, equally weighted for each class
        #class_weights = [1 / 3 for _ in range(3)]
        labels = [0, 1, 2]
        class_weights = [0.4, 0.2, 0.4]

        # Macro F1 is used when we care equally of each class, regardless of their frequency in the dataset
        f1 = f1_score(true_labels, predicted_labels, average='macro', labels=labels)
        #weighted_f1 = np.sum([w * f for w, f in zip(class_weights, f1)])
        return f1


