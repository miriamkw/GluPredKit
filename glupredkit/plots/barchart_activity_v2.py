import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from .base_plot import BasePlot
from sklearn.metrics import mean_squared_error
from glupredkit.helpers.unit_config_manager import unit_config_manager


# TODO Parse the activity log file (json format?)

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
'''

# after effect time version
# This version compares between RMSEafter 
class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):                
        y_pred = np.array(models_data[0].get('y_pred'))
        y_true = models_data[0].get('y_true')

        overall_rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        ph = models_data[0]['prediction_horizon']
        ph_step = ph // 5
        model_name = models_data[0]['name'].split(' ')[0]
        # print("model name data: ", model_name)
        effect_time_steps = [3, 6, 12, 18, 24] # activities lasting (15 min - 180 min)
        effect_minutes = [step * 5 for step in effect_time_steps]
        
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()

        after_rmses = []
        for i in range(len(effect_time_steps)):
            act_idx, act_true, act_pred = get_after_activity_data(y_true, y_pred, effect_time_steps[i])

            # Calculate RMSE per activity period
            rmses = []
            for j in range(len(act_true)):
                rmse = np.sqrt(mean_squared_error(act_true[j], act_pred[j]))
                rmses.append(rmse)
            after_rmses.append(rmses)
    
        print("after_rmses: ", after_rmses)
        bar_width = 1
        comparison_count = len(effect_time_steps)
        x_indices = np.arange(comparison_count)
        x_locations = np.arange(len(act_idx))
        colors = ['salmon', 'lightgreen', 'yellow', 'blue', 'red']
        rmse_per_comparison = []
        for i in range(len(act_idx)):
            for j in range(len(effect_time_steps)):
                rmse_per_comparison.append(after_rmses[j][i])
        
        after_rmses = np.array(after_rmses)
        x_label_locations = x_locations * comparison_count + bar_width * (comparison_count / 2) + x_locations
        labels = [str(step * 5) + ' minutes after PA' for step in effect_time_steps ]
        print("x_label_locations: ", x_label_locations)
        for idx, rmse_set in enumerate(after_rmses.T):
            if idx == 0: ax.bar(x_indices + idx * comparison_count + bar_width/2, rmse_set, width=bar_width, color=colors, alpha=0.7, label=labels)
            else: ax.bar(x_indices + idx + idx * comparison_count + bar_width/2, rmse_set, width=bar_width, color=colors, alpha=0.7)

        ax.axhline(y=overall_rmse, color='black', linestyle='--', label='Overall RMSE')
        plt.xticks(x_label_locations, ['A{}'.format(j + 1) for j in range(len(act_idx))])
        plt.title('{} RMSE after PA comparison (overall RMSE: %.2f)'.format(model_name) % overall_rmse)
        plt.tight_layout()
        plt.legend(loc='upper right')

        # Save the plot
        file_path = "data/figures/_ACTIVITY_COMPARISON/"
        config_str = sys.argv[3].replace(".pkl", "")
        file_name = f'Acvitivy_Trajectory_Comparison_{config_str}.png'
        save_plot(file_path, file_name)


    
def save_plot(file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(file_path + file_name)
    plt.show()


def get_after_activity_data(y_true, y_pred, effect_time_step):
    activity_period, activity_true, activity_pred = [], [], []
    prev_total_indices = None
    # Iterate through activity logs
    for log in activity_logs:
        original_start_time = pd.to_datetime(log['start_time'])
        duration = log['duration']
        after_time = original_start_time + pd.Timedelta(minutes=(duration* 5))
        after_effect_time = after_time + pd.Timedelta(minutes=(effect_time_step) * 5)

        # find indices within the activity period
        period_indices = (y_true.index >= after_time) & (y_true.index <= after_effect_time)
        period_indices = np.roll(period_indices, effect_time_step)

        # if(prev_total_indices is not None):
        #     overlap_period = np.logical_and(total_indices, prev_total_indices)
        #     if np.any(overlap_period):
        #         print("Overlap detected between total_indices and the next set of period_indices.")
        #         # TODO: handle the overlapping cases
        # prev_total_indices = total_indices

        activity_period.append(y_true.index[period_indices].strftime('%Y-%m-%d %H:%M:%S'))
        activity_true.append(np.array(y_true)[period_indices])
        activity_pred.append(np.array(y_pred)[period_indices])

    return [activity_period, activity_true, activity_pred]