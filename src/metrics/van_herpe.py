from src.metrics.base_metric import BaseMetric
from src.metrics.helper_functions import get_average_glucose_penalty_for_pairs
import numpy as np

class VanHerpe(BaseMetric):
    def __init__(self, is_bounded=False):
        super().__init__('VanHerpe')
        self.is_bounded = is_bounded

    def _calculate_metric(self, y_true, y_pred):
        def get_glucose_penalty(blood_glucose):
            if (self.is_bounded):
                if blood_glucose < 20:
                    return 100
                elif blood_glucose < 80:
                    return 7.4680 * (80 - blood_glucose)**0.6337
                elif blood_glucose <= 110:
                    return 0
                elif blood_glucose <= 250:
                    return 6.1767 * (blood_glucose - 110)**0.5635
                else:
                    return 100
            else:
                if blood_glucose < 80:
                    return 7.4680 * (80 - blood_glucose)**0.6337
                elif blood_glucose <= 110:
                    return 0
                else:
                    return 6.1767 * (blood_glucose - 110)**0.5635

        return get_average_glucose_penalty_for_pairs(y_true, y_pred, get_glucose_penalty)
    def set_is_bounded(self, value):
        self.is_bounded = value

