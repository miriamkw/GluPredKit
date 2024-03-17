import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from .base_plot import BasePlot
from sklearn.metrics import mean_squared_error
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


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        # TODO implement plot
        y_pred = models_data[0].get('y_pred')
        y_true = models_data[0].get('y_true')
        # print("models data: ", y_true.index)
        SHIFT_DURATION = 3 # 15 minutes
        rmse_period = []
        rmse_shifted = []
        activity_period = []
        idx = 0
        # Iterate through activity logs
        for log in activity_logs:
            idx += 1
            start_time = pd.to_datetime(log['start_time'])
            shifted_start_time = start_time + pd.Timedelta(minutes=(SHIFT_DURATION) * 5)

            duration = log['duration']
            end_time = start_time + pd.Timedelta(minutes=(duration) * 5)
            shifted_end_time = shifted_start_time + pd.Timedelta(minutes=(duration) * 5)
            
            # find indices within the activity period
            period_indices = (y_true.index >= start_time) & (y_true.index <= end_time)
            shifted_indices = (y_true.index >= shifted_start_time) & (y_true.index <= shifted_end_time)

            activity_period.append("Activity {}: {}".format(idx, log['activity']))
            
            # Calculate RMSE for the activity period
            rmse = np.sqrt(np.mean(np.square(np.array(y_true)[period_indices] - np.array(y_pred)[period_indices])))
            rmse_period.append(rmse)
            rmse = np.sqrt(np.mean(np.square(np.array(y_true)[shifted_indices] - np.array(y_pred)[shifted_indices])))
            rmse_shifted.append(rmse)

        # Plot RMSE for each activity period
        plt.figure(figsize=(10, 6))
        plt.plot(activity_period, rmse_period, label='RMSE during PA', marker='o', linestyle='')
        plt.plot(activity_period, rmse_shifted, label='RMSE shifted by 15 minutes', marker='x', linestyle='')

        plt.xlabel('Date')
        plt.ylabel('RMSE')
        plt.title('Comparison of RMSE during PA and RMSE shifted by 15 minutes')
        plt.xticks(rotation=60)
        plt.legend()
        plt.tight_layout()

        file_path = "data/figures/_ACTIVITY_COMPARISON/"
        os.makedirs(file_path, exist_ok=True)
        config_str = sys.argv[3].replace(".pkl", "")
        file_name = f'Acvitivy_RMSE_Comparison_{config_str}.png'
        plt.savefig(file_path + file_name)
        plt.show()

