from .base_metric import BaseMetric
import numpy as np
import pandas as pd
from glupredkit.helpers.unit_config_manager import unit_config_manager

# TODO Parse the activity log file (json format?)
'''
# perticipant 1
activity_logs = [{'start_time': "2024-01-26 21:00", 'duration': 12, 'activity': 'cross-country skiing'},
                 {'start_time': "2024-01-27 22:45", 'duration': 9, 'activity': 'biking'},
                 {'start_time': "2024-01-30 23:00", 'duration': 6, 'activity': 'strength training'},
                 {'start_time': "2024-01-31 20:15", 'duration': 9, 'activity': 'biking, walking, and snow shoveling'},
                 {'start_time': "2024-02-01 21:00", 'duration': 12, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-02 22:30", 'duration': 3, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-03 13:30", 'duration': 24, 'activity': 'cross-country skiing and snow-shoveling'},
                 {'start_time': "2024-02-03 20:00", 'duration': 12, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 13:00", 'duration': 6, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 23:30", 'duration': 6, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-05 22:00", 'duration': 6, 'activity': 'cross-country skiing'}
                 ]
'''
# participant 2
activity_logs = [{'start_time': "2024-02-12 09:30", 'duration': 6, 'activity': 'walk'},
                 {'start_time': "2024-02-12 12:30", 'duration': 6, 'activity': 'walk'},
                 {'start_time': "2024-02-12 16:25", 'duration': 4, 'activity': 'walk'},
                 {'start_time': "2024-02-12 19:55", 'duration': 3, 'activity': 'walk'},
                 {'start_time': "2024-02-12 23:10", 'duration': 4, 'activity': 'walk'},

                 {'start_time': "2024-02-13 07:50", 'duration': 4, 'activity': 'walk'},
                 {'start_time': "2024-02-13 09:50", 'duration': 4, 'activity': 'walk'},

                 {'start_time': "2024-02-14 14:15", 'duration': 4, 'activity': 'walk'},

                 {'start_time': "2024-02-15 07:50", 'duration': 2, 'activity': 'walk'},
                 {'start_time': "2024-02-15 13:40", 'duration': 6, 'activity': 'walk'},
                 {'start_time': "2024-02-15 17:40", 'duration': 4, 'activity': 'walk'},

                 {'start_time': "2024-02-17 17:00", 'duration': 8, 'activity': 'walk'},

                 {'start_time': "2024-02-18 17:30", 'duration': 6, 'activity': 'walk'},
                 {'start_time': "2024-02-18 22:00", 'duration': 6, 'activity': 'walk'}
                 ]

class Metric(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def __call__(self, y_true, y_pred, x):
        rmse = 0
        rmse_list = []

        # Iterate through activity logs
        for log in activity_logs:
            start_time = pd.to_datetime(log['start_time'])
            duration = log['duration']
            end_time = start_time + pd.Timedelta(minutes=(duration) * 5)
            # Find indices within the activity period
            indices = (y_true.index >= start_time) & (y_true.index <= end_time)
            # Calculate RMSE for the activity period
            ytrue = np.array(y_true)[indices]
            ypred = np.array(y_pred)[indices]
            rmse = np.sqrt(np.mean(np.square(ytrue - ypred)))

            # Normalization by Mean
            y_mean = np.mean(ytrue)
            rmse_mean_normalized = self.normalize_rmse_by_mean(rmse, y_mean)

            # Normalization by Difference Between Maximum and Minimum
            y_max = np.max(ytrue)
            y_min = np.min(ytrue)
            rmse_difference_normalized = self.normalize_rmse_by_difference(rmse, y_max, y_min)

            # Normalization by Standard Deviation
            y_std = np.std(ytrue)
            rmse_std_normalized = self.normalize_rmse_by_std(rmse, y_std)

            # Normalization by Interquartile Range
            q1, q3 = np.percentile(ytrue, [25, 75])
            rmse_iqr_normalized = self.normalize_rmse_by_iqr(rmse, q1, q3)

            rmses = {
                "RMSE": rmse,
                "RMSE_Mean_Normalized": rmse_mean_normalized,
                "RMSE_Difference_Normalized": rmse_difference_normalized,
                "RMSE_Std_Normalized": rmse_std_normalized,
                "RMSE_IQR_Normalized": rmse_iqr_normalized
            }
            rmse_list.append(rmses)

        if unit_config_manager.use_mgdl:
            return rmse_list

        
    def normalize_rmse_by_mean(self, rmse, y_mean):
        return rmse / y_mean
    
    def normalize_rmse_by_difference(self, rmse, y_max, y_min):
        return rmse / (y_max - y_min)
    
    def normalize_rmse_by_std(self, rmse, y_std):
        return rmse / y_std
    
    def normalize_rmse_by_iqr(self, rmse, q1, q3):
        return rmse / (q3 - q1)


