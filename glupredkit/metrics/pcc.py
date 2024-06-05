import numpy as np
from .base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self):
        super().__init__("PCC")

    def _calculate_metric(self, y_true, y_pred):
        # Convert y_true and y_pred to numpy arrays if they are not already
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Remove pairs with NaN values
        valid_indices = np.logical_not(np.logical_or(np.isnan(y_true), np.isnan(y_pred)))
        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]

        # Handle zero inputs
        if all(val == 0 for val in y_true_valid) or all(val == 0 for val in y_pred_valid):
            return 0.0  # Return 0 if all inputs are zero

        # Check if y_true_valid and y_pred_valid are empty or length of one
        if len(y_true_valid) <= 1:
            return 0.0  # Return 0 if the length is 1 or less

        corr_coef = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
        return corr_coef
