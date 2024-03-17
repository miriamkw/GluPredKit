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
        activity_period, activity_period_shifted, activity_period_total = [], [], []
        activity_true, activity_true_shifted, activity_pred, activity_pred_shifted, activity_true_total = [], [], [], [], []
        idx = 0
        # Iterate through activity logs
        for log in activity_logs:
            idx += 1
            start_time = pd.to_datetime(log['start_time'])

            duration = log['duration']
            end_time = start_time + pd.Timedelta(minutes=(duration) * 5)

            
            # find indices within the activity period
            period_indices = (y_true.index >= start_time) & (y_true.index <= end_time)
            shifted_indices = np.roll(period_indices, 3)
            total_indices = np.zeros(len(period_indices), dtype=bool)
            for idx in range(len(period_indices)):
                if shifted_indices[idx]:
                    total_indices[idx-3:idx+4] = True

            activity_period.append(y_true.index[period_indices].strftime('%Y-%m-%d %H:%M:%S'))
            activity_period_total.append(y_true.index[total_indices].strftime('%Y-%m-%d %H:%M:%S'))
            activity_period_shifted.append(y_true.index[shifted_indices].strftime('%Y-%m-%d %H:%M:%S'))

            # if(len(activity_period[idx-1]) != len(activity_period_shifted[idx-1])):
            #     print("Activity {}: length of activity period {}\n".format(idx, activity_period[idx-1]))
            #     print("Shifted Activity {}: length of shifted activity period {}\n".format(idx, activity_period_shifted[idx-1]))
                  
            activity_true.append(np.array(y_true)[period_indices])
            activity_pred.append(np.array(y_pred)[period_indices])
            activity_true_shifted.append(np.array(y_true)[shifted_indices])
            activity_pred_shifted.append(np.array(y_pred)[shifted_indices])
            activity_true_total.append(np.array(y_true)[total_indices])

        # Flatten the data into 1D arrays
        activity_period = [item for sublist in activity_period for item in sublist]
        activity_period_shifted = [item for sublist in activity_period_shifted for item in sublist]
        activity_true = [item for sublist in activity_true for item in sublist]
        activity_true_shifted = [item for sublist in activity_true_shifted for item in sublist]
        activity_pred = [item for sublist in activity_pred for item in sublist]
        activity_pred_shifted = [item for sublist in activity_pred_shifted for item in sublist]
        activity_period_total = [item for sublist in activity_period_total for item in sublist]
        activity_true_total = [item for sublist in activity_true_total for item in sublist]

        # Calculate MSE
        mse_pred = mean_squared_error(activity_true, activity_pred)
        mse_shift = mean_squared_error(activity_true_shifted, activity_pred_shifted)

        # Plot RMSE during PA
        plt.figure(figsize=(10, 6))
        plt.plot(activity_period_total, activity_true_total, color='k', alpha=0.9, label='Actual BG')
        plt.errorbar(activity_period, activity_pred, np.sqrt(mse_pred), color='red', alpha=0.6, label='RMSE during PA')
        plt.errorbar(activity_period_shifted, activity_pred_shifted, np.sqrt(mse_shift), color='blue', alpha=0.6, label='RMSE after 15 minutes of PA')
        plt.xlabel('Date')
        plt.ylabel('BG level (mg/dl)')
        plt.title('Comparison of RMSE during PA and RMSE shifted by 15 minutes')

        # Customizing x-axis and vertical lines in the plot
        ax = plt.gca() 
        pos = 0
        prev_datetime = pd.to_datetime(activity_period_total[0]) 
        x_labels = [activity_period_total[0]] 
        for date_time in activity_period_total[1:]:
            current_datetime = pd.to_datetime(date_time)
            time_diff = (current_datetime - prev_datetime).total_seconds() / 60  # time diff in min
            if time_diff > 5:
                x_labels.append(date_time)
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
            else:
                x_labels.append('')  # Empty string for time gaps less than 5 minutes
            prev_datetime = current_datetime
            pos += 1

        plt.xticks(range(len(activity_period_total)), x_labels, rotation=90)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        file_path = "data/figures/_ACTIVITY_COMPARISON/"
        os.makedirs(file_path, exist_ok=True)
        config_str = sys.argv[3].replace(".pkl", "")
        file_name = f'Acvitivy_RMSE_Error_Grid_Comparison_{config_str}.png'
        plt.savefig(file_path + file_name)
        plt.show()

