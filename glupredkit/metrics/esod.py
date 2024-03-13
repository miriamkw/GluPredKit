from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager

'''
The ideal ESODn is one, and the closer to one, the better the predicted time series are considered. 
'''

class Metric(BaseMetric):
    def __init__(self):
        super().__init__('ESOD')

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        numerator = denominator = 0
        # Calculate the energy of the second-order differences
        for k in range(2, len(y_true)):
            numerator += np.sum((y_pred[k] - 2*y_pred[k-1] + y_pred[k-2])**2)
            denominator += np.sum((y_true[k] - 2*y_true[k-1] + y_true[k-2])**2)

        # Compute the normalized energy (ESODn)
        if denominator != 0:
            ESODn = numerator / denominator
        else: 
            raise Exception("Denominator is zero")

        return ESODn


