from .base_metric import BaseMetric
from error_grids import zone_accuracy


from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Metric(BaseMetric):
    def __init__(self, target=112, bg_normalization_factor=4):
        super().__init__('Continous SEG')
        self.target = target
        self.bg_normalization_factor = bg_normalization_factor

    def _calculate_metric(self, y_true, y_pred, use_normalization=False):
        if use_normalization:
            # Normalize glucose values
            observed_values = self.normalize_bg(y_true)
            predicted_values = self.normalize_bg(y_pred)
            target = self.normalize_bg(self.target)
        else:
            observed_values = y_true
            predicted_values = y_pred
            target = np.array(self.target)

        # Amount of needed correction (in terms of normalized bg) for predicted and observed
        correction = target - predicted_values
        correction_ref = target - observed_values

        # Difference in bg correction
        correction_diff = correction - correction_ref

        # Scale the difference relative to reference correction, but avoid division by zero by using numpy where
        scaled_correction = np.where(correction_ref != 0, correction_diff / correction_ref, 1)

        # Risk index
        risk = np.abs(np.clip(scaled_correction, -1, 1))

        return np.average(risk)

    def normalize_bg(self, x):
        x = np.array(x)  # Ensure x is a numpy array
        return np.power(np.log(x / 600 + 1), 1.0 / self.bg_normalization_factor)

    def denormalize_bg(self, x):
        return (np.exp(np.power(x, self.bg_normalization_factor)) - 1) * 600


