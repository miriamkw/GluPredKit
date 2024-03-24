"""
Hyperglycaemia detection calculates how many percentages of the predictions predicts a value above 180 mg/dL when
the target value is above 180 mg/dL.
"""
from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Hyperglycemia Detection')

    def __call__(self, y_true, y_pred):
        # Filter out indices where y_true values are above 10
        filtered_indices = [i for i, val in enumerate(y_true) if val >= 180]
        filtered_y_true = [y_true[i] for i in filtered_indices]
        filtered_y_pred = [y_pred[i] for i in filtered_indices]

        # Calculate the total number of instances where both filtered y_true and filtered y_pred are above 10
        total_instances = sum(
            1 for true_val, pred_val in zip(filtered_y_true, filtered_y_pred) if true_val > 180 and pred_val > 180)

        # Calculate the total number of instances in the filtered data
        total_filtered_instances = len(filtered_y_true)

        # Calculate the percentage
        if total_filtered_instances > 0:
            percentage = (total_instances / total_filtered_instances) * 100
        else:
            percentage = 0

        return percentage
