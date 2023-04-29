from src.metrics.base_metric import BaseMetric
from src.metrics.helper_functions import get_average_glucose_penalty_for_pairs
import numpy as np

class Bayer(BaseMetric):
    def __init__(self, target=105):
        super().__init__('Bayer')
        self.target = target

    def __call__(self, y_true, y_pred):
        def get_glucose_penalty(blood_glucose):
            blood_glucose = max(blood_glucose, 1)
            return 32.9170208165394 * (np.log(blood_glucose / self.target))**2

        return get_average_glucose_penalty_for_pairs(y_true, y_pred, get_glucose_penalty)

    def set_target(self, value):
        self.target = value
