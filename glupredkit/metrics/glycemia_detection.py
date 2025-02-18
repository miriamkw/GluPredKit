"""
Glycemia detection calculates the confusion matrix. The metric returns two lists, the first containing the percentages
in each region, while the second returns the number of samples.
"""
from .base_metric import BaseMetric
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Glycemia Detection')
        self.hypo_threshold = 70
        self.hyper_threshold = 180

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Filter out invalid pairs
        filtered_pairs = [(x, y) for x, y in zip(y_true, y_pred) if np.isfinite(x) and np.isfinite(y)]

        # Ensure filtered_pairs is not empty
        if not filtered_pairs:
            print("No valid pairs of true and predicted values!")
            return np.full((3, 3), np.nan)

        # Remove nan measurements or predictions
        y_true, y_pred = map(
            np.array,
            zip(*[(x, y) for x, y in zip(y_true, y_pred) if np.isfinite(x) and np.isfinite(y)])
        )

        # Unpack the filtered pairs
        y_true, y_pred = map(np.array, zip(*filtered_pairs))

        percentages = []
        n_conditions = 3

        for i in range(n_conditions):
            for j in range(n_conditions):
                true_count = np.sum(self.condition(y_true, j))
                relevant_indices = np.where(self.condition(y_true, j))
                relevant_predictions = y_pred[relevant_indices]
                pred_count = np.sum(self.condition(relevant_predictions, i))

                if true_count > 0:
                    percentage = pred_count / true_count
                    percentages += [percentage]
                else:
                    percentages += [np.nan]

        matrix = np.array(percentages).reshape(3, 3).tolist()
        return matrix

    def condition(self, x, condition):
        if condition == 0:
            return x < self.hypo_threshold
        elif condition == 1:
            return (x >= self.hypo_threshold) & (x <= self.hyper_threshold)
        else:
            return x > self.hyper_threshold

