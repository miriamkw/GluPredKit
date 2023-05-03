from src.metrics.base_metric import BaseMetric
from src.metrics.helper_functions import get_average_glucose_penalty_for_pairs

class Cao(BaseMetric):
    def __init__(self, is_bounded=False):
        super().__init__('Cao')
        self.is_bounded = is_bounded

    def _calculate_metric(self, y_true, y_pred):
        def get_glucose_penalty(blood_glucose):
            if self.is_bounded:
                if blood_glucose < 50:
                    return 100
                elif blood_glucose <= 80:
                    return 1.0567 * (80 - blood_glucose)**1.3378
                elif blood_glucose <= 140:
                    return 0
                elif blood_glucose <= 300:
                    return 0.4607 * (blood_glucose - 140)**1.0601
                else:
                    return 100
            else:
                if blood_glucose <= 80:
                    return 1.0567 * (80 - blood_glucose)**1.3378
                elif blood_glucose <= 140:
                    return 0
                else:
                    return 0.4607 * (blood_glucose - 140)**1.0601

        return get_average_glucose_penalty_for_pairs(y_true, y_pred, get_glucose_penalty)

    def set_is_bounded(self, value):
        self.is_bounded = value
