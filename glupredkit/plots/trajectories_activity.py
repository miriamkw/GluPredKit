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
        ph = models_data[0]['prediction_horizon']
        ph_step = ph // 5
        model_name = models_data[0]['name'].split(' ')[0]
        # print("model name data: ", model_name)
        SHIFT_DURATION = 3 # 15 minutes
        rmse_period = []
        rmse_shifted = []
        activity_period, activity_period_shifted, activity_period_total, activity_period_shifted_total = [], [], [], []
        activity_true, activity_true_shifted, activity_pred, activity_pred_shifted, activity_true_total, activity_true_shifted_total = [], [], [], [], [], []
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
            total_indices = np.zeros(len(period_indices + ph_step), dtype=bool)
            for idx in range(len(period_indices)):
                if period_indices[idx]:
                    total_indices[idx:idx+13] = True

            activity_period.append(y_true.index[period_indices].strftime('%Y-%m-%d %H:%M:%S'))
            activity_period_total.append(y_true.index[total_indices].strftime('%Y-%m-%d %H:%M:%S'))
            activity_period_shifted.append(y_true.index[shifted_indices].strftime('%Y-%m-%d %H:%M:%S'))
            activity_period_shifted_total.append(y_true.index[np.roll(total_indices, 3)].strftime('%Y-%m-%d %H:%M:%S'))

            activity_true.append(np.array(y_true)[period_indices])
            activity_pred.append(np.array(y_pred)[period_indices])
            activity_true_shifted.append(np.array(y_true)[shifted_indices])
            activity_pred_shifted.append(np.array(y_pred)[shifted_indices])
            activity_true_total.append(np.array(y_true)[total_indices])
            activity_true_shifted_total.append(np.array(y_true)[np.roll(total_indices, 3)])

        # Calculate RMSE per activity
        activity_traj = activity_period
        acitvity_shifted_traj = activity_period_shifted
        activity_rmse, activity_shifted_rmse = [], []
        for i in range(len(activity_true)):
            rmse = np.sqrt(mean_squared_error(activity_true[i], activity_pred[i]))
            activity_rmse.append(rmse)

        for i in range(len(activity_true_shifted)):
            rmse = np.sqrt(mean_squared_error(activity_true_shifted[i], activity_pred_shifted[i]))
            activity_shifted_rmse.append(rmse)
            #print("Activity {}: RMSE during PA: {}".format(i, rmse))

        # Flatten the data into 1D arrays
        activity_period = [item for sublist in activity_period for item in sublist]
        activity_period_shifted = [item for sublist in activity_period_shifted for item in sublist]
        activity_true = [item for sublist in activity_true for item in sublist]
        activity_true_shifted = [item for sublist in activity_true_shifted for item in sublist]
        activity_pred = [item for sublist in activity_pred for item in sublist]
        activity_pred_shifted = [item for sublist in activity_pred_shifted for item in sublist]
        activity_period_total = [item for sublist in activity_period_total for item in sublist]
        activity_period_shifted_total = [item for sublist in activity_period_shifted_total for item in sublist]
        activity_true_total = [item for sublist in activity_true_total for item in sublist]
        activity_true_shifted_total = [item for sublist in activity_true_shifted_total for item in sublist]

        # Plot RMSE during PA
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(activity_period_total, activity_true_total, color='k', alpha=0.9, label='Actual BG')
        
        # Add predicted trajectories during the PA
        add_predicted_traectories(plt, activity_traj, y_true, y_pred, ph, model_name)        
        plt.title('RMSE during PA')

        # Customizing x-axis and vertical lines in the plot
        label_and_divide_plot(plt, activity_period_total, activity_rmse)
        
        # Second plot for shited version
        plt.subplot(1, 2, 2)
        plt.plot(activity_period_shifted_total, activity_true_shifted_total, color='k', alpha=0.9, label='Actual BG')

        # Add predicted trajectories 15 min after the start of PA (shifted)
        add_predicted_traectories(plt, acitvity_shifted_traj, y_true, y_pred, ph, model_name) 
        plt.title('RMSE after 15 minutes from start of PA')

        # Customizing x-axis and vertical lines in the plot
        label_and_divide_plot(plt, activity_period_shifted_total, activity_shifted_rmse)
        
        plt.tight_layout()

        # Save the plot
        file_path = "data/figures/_ACTIVITY_COMPARISON/"
        config_str = sys.argv[3].replace(".pkl", "")
        file_name = f'Acvitivy_Trajectory_Comparison_{config_str}.png'
        save_plot(file_path, file_name)



def plot_activity_comparison(activity_period_total, activity_true_total, activity_pred_total, rmse_values, model_name, title):
    plt.plot(activity_period_total, activity_true_total, color='k', alpha=0.9, label='Actual BG')
    for i in range(len(activity_period_total)):
        plt.plot(activity_period_total[i], activity_pred_total[i], linestyle='--', label=f'Predicted trajectory - {model_name}')
        plt.text(i, 180, '{}'.format(round(rmse_values[i], 2)), verticalalignment='top')
    plt.xlabel('Date')
    plt.ylabel('BG level (mg/dL)')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()


    
def save_plot(file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(file_path + file_name)
    plt.show()


def add_predicted_traectories(plt, activity_trajectory_index, actual, predicted, ph, model_name):
    ph_step = ph // 5
    for i in range(len(activity_trajectory_index)):
        trajectory = []
        t_values = []
        trajectory_index, trajectory_lines = [], []

        t_values = pd.to_datetime(activity_trajectory_index[i])
        t_values_indices = [actual.index.get_loc(t) for t in t_values if t in actual.index]
        trajectory = predicted[t_values_indices]

        for k in range(0, len(trajectory)):
            trajectory_index = [t_values[k].strftime('%Y-%m-%d %H:%M:%S'), (t_values[k] + pd.Timedelta(minutes=ph)).strftime('%Y-%m-%d %H:%M:%S')]
            if t_values_indices[0] + k < len(predicted) and t_values_indices[0] + k + ph_step < len(predicted):
                trajectory_lines = [actual[t_values_indices[0] + k], predicted[t_values_indices[0] + k + ph_step]]

            if trajectory_index and trajectory_lines:
                plt.plot(trajectory_index, trajectory_lines, linestyle='--', label=f'Predicted trajectory - {model_name}')
    plt.xlabel('Date')
    plt.ylabel('BG level (mg/dL)')


def label_and_divide_plot(plt, act_period_total, act_rmse):
    ax = plt.gca() 
    ax.set_ylim(20, 290)
    pos = counter = prev_pos = 0
    prev_datetime = pd.to_datetime(act_period_total[0]) 
    x_labels = [act_period_total[0]] 

    for date_time in act_period_total[1:]:
        current_datetime = pd.to_datetime(date_time)
        time_diff = (current_datetime - prev_datetime).total_seconds() / 60  # time diff in min
        temp_rmse = act_rmse[counter]
            
        if time_diff > 5:
            x_labels.append(date_time)
            plt.text(prev_pos, 280, '{}'.format(round(temp_rmse,2)), verticalalignment='top')
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
            counter += 1
            prev_pos = pos
        else:
            x_labels.append('')  # Empty string for time gaps less than 5 minutes
        prev_datetime = current_datetime
        pos += 1

    plt.xticks(range(len(act_period_total)), x_labels, rotation=90)