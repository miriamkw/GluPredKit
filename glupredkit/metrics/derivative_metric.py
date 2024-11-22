import numpy as np
from .base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self):
        super().__init__("Derivative Metric")

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        # Convert y_true and y_pred to numpy arrays if they are not already
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Remove pairs with NaN values
        valid_indices = np.logical_not(np.logical_or(np.isnan(y_true), np.isnan(y_pred)))
        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]

        # Handle case where there's too little data
        if len(y_true_valid) <= 1:
            return 0.0  # Return 0 if the length is 1 or less

        # Compute the first-order differences (slopes) for both true and predicted values
        true_derivatives = np.diff(y_true_valid)
        pred_derivatives = np.diff(y_pred_valid)

        # Calculate the absolute difference between the slopes
        derivative_error = np.abs(true_derivatives - pred_derivatives)

        # We can return the mean error as the metric (smaller is better)
        return np.mean(derivative_error)

