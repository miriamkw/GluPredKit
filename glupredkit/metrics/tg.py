from .base_metric import BaseMetric
import numpy as np
from glupredkit.helpers.unit_config_manager import unit_config_manager
import sys

# Aim: Calculate delay (delay(x, ^x)) by finding the temporal shift that minimizes the distance between x and ^x
# A larger TG implies an earlier detection of a potential hypo/hyper glycemia event. A zero-order-hold prediction will render T G = 0 and thus is not useful from a clinical perspective
class Metric(BaseMetric):
    def __init__(self):
        pred_horizon = sys.argv[3].split("_")[-1].replace(".pkl","")
        super().__init__('TG')
        self.PH = int(pred_horizon) // 5
        self.delta_t = 5 # measured every 5 minutes

    def __call__(self, y_true, y_pred):
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

