from .base_metric import BaseMetric
import numpy as np
import pandas as pd
from glupredkit.helpers.unit_config_manager import unit_config_manager
from statsmodels.tsa.arima.model import ARIMA
import sys
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
        pred_horizon = sys.argv[3].split("_")[-1].replace(".pkl","")
        self.PH = int(pred_horizon) // 5

    def __call__(self, y_true, y_pred, test_data):
        rmse = 0
        rmse_list = []
        p, d, q = 24, 1, 0
        history = np.array(test_data['CGM'])

        # Iterate through activity logs
        for log in activity_logs:
            start_time = pd.to_datetime(log['start_time'])
            history_start_time = start_time - pd.Timedelta(minutes=(self.PH) * 5)
            duration = log['duration']
            end_time = start_time + pd.Timedelta(minutes=(duration) * 5)
            history_end_time = history_start_time + pd.Timedelta(minutes=(duration) * 5)

            # Find indices within the activity period
            history_indices = y_true.index < history_start_time
            previous_indices = (y_true.index >= history_start_time)
            indices = (y_true.index >= start_time) & (y_true.index <= end_time)

            # ARIMA forecasts during the activity period
            # version1 uses the total previous data before the time of forecast
            forecasts = []
            history = np.append(history, np.array(y_true)[history_indices])
            previous_values = np.array(y_true)[previous_indices]
            for i in range(len(np.array(y_true)[indices])):
                # Train ARIMA
                model = ARIMA(history, order=(p, d, q))
                model_fit = model.fit()
                output = model_fit.forecast(steps=self.PH)
                yhat = output[-1]
                forecasts.append(yhat)
                observation = previous_values[i] 
                history = np.append(history, observation)

            print("np.array(y_true)[indices]: ", np.array(y_true)[indices])
            print("np.array(forecasts): ", np.array(forecasts))
            # Calculate RMSE for the activity period
            rmse = np.sqrt(np.mean(np.square(np.array(y_true)[indices] - np.array(forecasts))))
            
            rmse_list.append(rmse)

        # avg_rmse = rmse / count
        
        if unit_config_manager.use_mgdl:
            return rmse_list
        else:
            return [unit_config_manager.convert_value(rmse) for rmse in rmse_list]

