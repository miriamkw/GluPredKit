from .base_metric import BaseMetric
import numpy as np
import pandas as pd
from glupredkit.helpers.unit_config_manager import unit_config_manager

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

    def __call__(self, y_true, y_pred):
        rmse = 0
        # count = 0
        rmse_list = []

        # Iterate through activity logs
        for log in activity_logs:
            start_time = pd.to_datetime(log['start_time'])
            duration = log['duration']
            print("start time: ", start_time)
            # Calculate end time
            end_time = start_time + pd.Timedelta(minutes=duration * 5)
            print("end time: ", end_time)
            # Find indices within the activity period
            indices = (y_true.index >= start_time) & (y_true.index <= end_time)
            print("indices: ", indices)
            # Calculate RMSE for the activity period
            rmse = np.sqrt(np.mean(np.square(np.array(y_true)[indices] - np.array(y_pred)[indices])))
            print("np.array(y_true)[indices]: ", np.array(y_true)[indices])
            print("np.array(y_pred)[indices]: ", np.array(y_pred)[indices])
            rmse_list.append(rmse)

        # avg_rmse = rmse / count
        
        if unit_config_manager.use_mgdl:
            return rmse_list
        else:
            return [unit_config_manager.convert_value(rmse) for rmse in rmse_list]
