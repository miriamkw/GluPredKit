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



# after effect time version

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
        effect_time_steps = [3, 6, 12, 18] # activities lasting (15 min - 90 min)
        effect_minutes = [step * 5 for step in effect_time_steps]
        
        plt.figure(figsize=(18, 10))

        for i in range(len(effect_time_steps)):
            act_idx, act_idx_total, act_true, act_pred, act_true_total = get_after_activity_data(y_true, y_pred, effect_time_steps[i], ph_step)
            print("act_idx: ", act_idx)
            # print("act_idx_total: ", act_idx_total)
            # print("act_true: ", act_true)
            # print("act_pred: ", act_pred)
            # print("act_true_total: ", act_true_total)
            activity_traj = act_idx # save it before flattening the data
            # Calculate RMSE per activity period
            activity_rmse = []
            for j in range(len(act_true)):
                rmse = np.sqrt(mean_squared_error(act_true[j], act_pred[j]))
                activity_rmse.append(rmse)

            # Flatten the data into 1D arrays
            act_idx = [item for sublist in act_idx for item in sublist]
            act_true = [item for sublist in act_true for item in sublist]
            act_pred = [item for sublist in act_pred for item in sublist]
            act_idx_total = [item for sublist in act_idx_total for item in sublist]
            act_true_total = [item for sublist in act_true_total for item in sublist]

            # row = 1 if i // 2 < 1 else 2
            plt.subplot(2, 2, (i + 1))
            plt.plot(act_idx_total, act_true_total, color='k', alpha=0.9, label='Actual BG')
            # Add predicted trajectories during the PA
            add_predicted_trajectories(plt, activity_traj, y_true, y_pred, ph, model_name)        
            plt.title('RMSE after PA {} min (overall RMSE: %.2f)'.format(str(effect_minutes[i])) % overall_rmse)

            # Customizing x-axis and vertical lines in the plot
            label_and_divide_plot(plt, act_idx_total, activity_rmse)
   
        plt.tight_layout()

        # Save the plot
        file_path = "data/figures/_ACTIVITY_COMPARISON/"
        config_str = sys.argv[3].replace(".pkl", "")
        file_name = f'Acvitivy_Trajectory_Comparison_{config_str}.png'
        save_plot(file_path, file_name)


'''
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
'''

    
def save_plot(file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(file_path + file_name)
    plt.show()


def add_predicted_trajectories(plt, activity_trajectory_index, actual, predicted, ph, model_name):
    ph_step = ph // 5
    print("activity_trajectory_index: ", activity_trajectory_index)
    for i in range(len(activity_trajectory_index)):
        trajectory = []
        t_values = []
        trajectory_index, trajectory_lines = [], []

        t_values = pd.to_datetime(activity_trajectory_index[i])
        t_values_indices = [actual.index.get_loc(t) for t in t_values if t in actual.index]
        print("predicted: ", predicted)
        trajectory = predicted[t_values_indices]

        for k in range(0, len(trajectory)):
            trajectory_index = [t_values[k].strftime('%Y-%m-%d %H:%M:%S'), (t_values[k] + pd.Timedelta(minutes=ph)).strftime('%Y-%m-%d %H:%M:%S')]
            if t_values_indices[0] + k < len(predicted) and t_values_indices[0] + k + ph_step < len(predicted):
                trajectory_lines = [actual[t_values_indices[0] + k], predicted[t_values_indices[0] + k + ph_step]]

            if trajectory_index and trajectory_lines:
                plt.plot(trajectory_index, trajectory_lines, linestyle='--', label=f'Predicted trajectory - {model_name}')
    #plt.xlabel('Date - Activity time and after effect time comparison')
    plt.ylabel('BG level (mg/dL)')


def label_and_divide_plot(plt, act_period_total, act_rmse):
    ax = plt.gca() 
    ax.set_ylim(20, 350)
    pos = counter = prev_pos = 0
    prev_datetime = pd.to_datetime(act_period_total[0]) 
    x_labels = [act_period_total[0]] 

    for date_time in act_period_total[1:]:
        current_datetime = pd.to_datetime(date_time)
        time_diff = (current_datetime - prev_datetime).total_seconds() / 60  # time diff in min
        temp_rmse = act_rmse[counter]
            
        if time_diff > 5:
            x_labels.append(date_time)
            plt.text(prev_pos, 310, '{}'.format(round(temp_rmse,2)), verticalalignment='top')
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
            ax.axhspan(70, 180, color='yellow', alpha=0.3) # healthy range
            counter += 1
            prev_pos = pos
        else:
            x_labels.append('')  # Empty string for time gaps less than 5 minutes
        prev_datetime = current_datetime
        pos += 1

    plt.xticks(range(len(act_period_total)), x_labels, rotation=90)

'''
# shift version
def get_activity_logs(y_true, y_pred, effect_time_step, ph_step):
    activity_period, activity_period_total, activity_true, activity_pred, activity_true_total = [], [], [], [], []
    prev_total_indices = None
    # Iterate through activity logs
    
    for log in activity_logs:
        start_time = pd.to_datetime(log['start_time'])

        duration = log['duration']
        end_time = start_time + pd.Timedelta(minutes=(duration + effect_time_step)  * 5)

        # find indices within the activity period
        period_indices = (y_true.index >= start_time) & (y_true.index <= end_time)
        period_indices = np.roll(period_indices, effect_time_step)
        total_indices = np.zeros(len(period_indices + ph_step), dtype=bool)

        for idx in range(len(period_indices)):
            if period_indices[idx]:
                total_indices[idx:idx + ph_step + 1] = True # more hours (buffer) for trajectory

        # if(prev_total_indices is not None):
        #     overlap_period = np.logical_and(total_indices, prev_total_indices)
        #     if np.any(overlap_period):
        #         print("Overlap detected between total_indices and the next set of period_indices.")
        #         # TODO: handle the overlapping cases
        # prev_total_indices = total_indices

        activity_period.append(y_true.index[period_indices].strftime('%Y-%m-%d %H:%M:%S'))
        activity_period_total.append(y_true.index[total_indices].strftime('%Y-%m-%d %H:%M:%S'))
        activity_true.append(np.array(y_true)[period_indices])
        activity_pred.append(np.array(y_pred)[period_indices])
        activity_true_total.append(np.array(y_true)[total_indices])

    return [activity_period, activity_period_total, activity_true, activity_pred, activity_true_total]
'''

# after effect version
def get_after_activity_data(y_true, y_pred, effect_time_step, ph_step):
    activity_period, activity_period_total, activity_true, activity_pred, activity_true_total = [], [], [], [], []
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
        total_indices = np.zeros(len(period_indices + ph_step), dtype=bool)

        for idx in range(len(period_indices)):
            if period_indices[idx]:
                total_indices[idx:idx + ph_step + 1] = True # more hours (buffer) for trajectory

        # if(prev_total_indices is not None):
        #     overlap_period = np.logical_and(total_indices, prev_total_indices)
        #     if np.any(overlap_period):
        #         print("Overlap detected between total_indices and the next set of period_indices.")
        #         # TODO: handle the overlapping cases
        # prev_total_indices = total_indices

        activity_period.append(y_true.index[period_indices].strftime('%Y-%m-%d %H:%M:%S'))
        activity_period_total.append(y_true.index[total_indices].strftime('%Y-%m-%d %H:%M:%S'))
        activity_true.append(np.array(y_true)[period_indices])
        activity_pred.append(np.array(y_pred)[period_indices])
        activity_true_total.append(np.array(y_true)[total_indices])

    return [activity_period, activity_period_total, activity_true, activity_pred, activity_true_total]