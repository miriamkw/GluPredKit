from .base_metric import BaseMetric
import numpy as np
import pandas as pd
from glupredkit.helpers.unit_config_manager import unit_config_manager

'''
# Eirik's
activity_logs = [{'start_time': "2024-02-03 13:30", 'duration': 2, 'activity': 'cross-country skiing and snow-shoveling'},
                 {'start_time': "2024-02-03 20:00", 'duration': 1, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 13:00", 'duration': 0.5, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 23:30", 'duration': 0.5, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-05 22:00", 'duration': 0.5, 'activity': 'cross-country skiing'}
                 ]
'''
activity_logs = [{'start_time': "2024-02-17 17:00", 'duration': 0.67, 'activity': 'walk'},
                 {'start_time': "2024-02-18 17:30", 'duration': 0.5, 'activity': 'walk'},
                 {'start_time': "2024-02-18 22:00", 'duration': 0.5, 'activity': 'walk'}
                 ]

class Metric(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def __call__(self, y_true, y_pred):
        rmse = 0
        # count = 0
        rmse_list = []

        # Iterate through activity logs
        for log in activity_logs:
            start_time = pd.to_datetime(log['start_time'])
            duration = log['duration']

            # Calculate end time
            end_time = start_time + pd.Timedelta(hours=duration)

            # Find indices within the activity period
            indices = (y_true.index >= start_time) & (y_true.index <= end_time)

            # Calculate RMSE for the activity period
            rmse = np.sqrt(np.mean(np.square(np.array(y_true)[indices] - np.array(y_pred)[indices])))
            rmse_list.append(rmse)

        # avg_rmse = rmse / count
        
        if unit_config_manager.use_mgdl:
            return rmse_list
        else:
            return [unit_config_manager.convert_value(rmse) for rmse in rmse_list]

