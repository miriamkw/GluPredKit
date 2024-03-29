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
        y_pred = models_data[0].get('y_pred')
        y_true = models_data[0].get('y_true')

        overall_rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))

        activity_true, activity_pred = [], []
        idx = 0
        # Iterate through activity logs
        for log in activity_logs:
            idx += 1
            start_time = pd.to_datetime(log['start_time'])
            duration = log['duration']
            end_time = start_time + pd.Timedelta(minutes=(duration) * 5)

            # find indices within the activity period
            period_indices = (y_true.index >= start_time) & (y_true.index <= end_time)
                  
            activity_true.append(np.array(y_true)[period_indices])
            activity_pred.append(np.array(y_pred)[period_indices])
        
        rmse_values = []
        for i in range(len(activity_true)):
            rmse = np.sqrt(mean_squared_error(activity_true[i], activity_pred[i]))
            rmse_values.append(rmse)
        # Calculate summary statistics
        mean_rmse = np.mean(rmse_values)
        median_rmse = np.median(rmse_values)
        std_rmse = np.std(rmse_values)
        min_rmse = np.min(rmse_values)
        max_rmse = np.max(rmse_values)

        print("Mean RMSE:", mean_rmse)
        print("Median RMSE:", median_rmse)
        print("Standard Deviation of RMSE:", std_rmse)
        print("Minimum RMSE:", min_rmse)
        print("Maximum RMSE:", max_rmse)

        config_str = sys.argv[3].replace(".pkl", "")
        model_name = config_str.split("__")[0].replace("_", " ").upper()
        # Create box plot of RMSE values
        plt.figure(figsize=(4, 4))
        plt.boxplot(rmse_values)
        plt.xlabel(model_name)
        plt.ylabel('RMSE (mg/dL)')
        plt.grid(True)
        plt.text(1, mean_rmse, f'Mean: {mean_rmse:.2f}', bbox=dict(facecolor='white', alpha=0.5))
        # plt.text(1, max_rmse + 0.1, f'Overall RMSE: {mean_rmse:.2f}', ha='center', va='center', 
        #          bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))

        file_path = "data/figures/_RMSE_VARIABILITY/"
        file_name = f'RMSE_variability_{config_str}.png'

        if len(sys.argv) < 7:
            plt.title(f'Box Plot of RMSE During PA Periods\nOverall RMSE: {overall_rmse:.2f}', fontsize=10)
            save_plot(file_path, file_name)
        else:
            plt.title(f'Overall RMSE: {overall_rmse:.2f}', fontsize=10)
            os.makedirs(file_path, exist_ok=True)
            plt.savefig(file_path + file_name)



def save_plot(file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(file_path + file_name)
    plt.show()