from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager
import sys

class Metric(BaseMetric):
    def __init__(self):
        pred_horizon = sys.argv[3].split("_")[6].replace(".pkl","")
        super().__init__('JINDEX')
        self.PH = int(pred_horizon) // 5
        self.delta_t = 5 # measured every 5 minutes

    def __call__(self, y_true, y_pred):
        ESODn = self._get_ESODn(y_pred, y_true)
        TG = self._get_TG(y_pred, y_true)

        J_idx = ESODn/(TG/self.PH)
        return J_idx
    
    def _get_TG(self, y_pred, y_true):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Initialize minimum sum of squared differences and optimal delay
        min_sum_squared_err = float('inf')
        optimal_delay = 0
        N = len(y_true)
        for i in range(self.PH + 1):  # i can range from 0 to L
            # Calculate the sum of squared differences for the current delay (i)
            sum_squared_err = sum((y_pred[k + i] - y_true[k])**2 for k in range(1, N  - self.PH)) / (N  - self.PH)
            
            # Update the minimum sum of squared differences and optimal delay if needed
            if sum_squared_err < min_sum_squared_err:
                min_sum_squared_err = sum_squared_err
                optimal_delay = i
        
        # Calculate temporal gain using the delay and delta_t
        TG = (self.PH - optimal_delay) * self.delta_t

        return TG
    
    def _get_ESODn(self, y_pred, y_true):
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