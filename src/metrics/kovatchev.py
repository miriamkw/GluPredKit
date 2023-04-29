from src.metrics.base_metric import BaseMetric
from src.metrics.helper_functions import get_average_glucose_penalty_for_pairs
import numpy as np

class Kovatchev(BaseMetric):
    def __init__(self):
        super().__init__('Kovatchev')

    def __call__(self, y_true, y_pred):
        def get_glucose_penalty(blood_glucose):
            blood_glucose = max(blood_glucose, 1)
            return 10 * (1.509 * (np.log(blood_glucose)**1.084 - 5.381))**2

        return get_average_glucose_penalty_for_pairs(y_true, y_pred, get_glucose_penalty)


