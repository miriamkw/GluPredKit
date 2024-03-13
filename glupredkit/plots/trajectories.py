import datetime
from matplotlib.ticker import LinearLocator
from .base_plot import BasePlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glupredkit.helpers.cli as helpers
from glupredkit.helpers.unit_config_manager import unit_config_manager
import os


# participant 1
activity_logs = [{'start_time': "2024-01-26 21:00", 'duration': 1, 'activity': 'cross-country skiing'},
                 {'start_time': "2024-01-27 22:45", 'duration': 0.75, 'activity': 'biking'},
                 {'start_time': "2024-01-30 23:00", 'duration': 0.5, 'activity': 'strength training'},
                 {'start_time': "2024-01-31 20:15", 'duration': 0.75, 'activity': 'biking, walking, and snow shoveling'},
                 {'start_time': "2024-02-01 21:00", 'duration': 1, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-02 22:30", 'duration': 0.25, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-03 13:30", 'duration': 2, 'activity': 'cross-country skiing and snow-shoveling'},
                 {'start_time': "2024-02-03 20:00", 'duration': 1, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 13:00", 'duration': 0.5, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-04 23:30", 'duration': 0.5, 'activity': 'snow-shoveling'},
                 {'start_time': "2024-02-05 22:00", 'duration': 0.5, 'activity': 'cross-country skiing'}
                 ]
'''
# participant 2
activity_logs = [{'start_time': "2024-02-12 09:30", 'duration': 0.5, 'activity': 'walk'},
                 {'start_time': "2024-02-12 12:30", 'duration': 0.5, 'activity': 'walk'},
                 {'start_time': "2024-02-12 16:25", 'duration': 0.3, 'activity': 'walk'},
                 {'start_time': "2024-02-12 19:55", 'duration': 0.25, 'activity': 'walk'},
                 {'start_time': "2024-02-12 23:10", 'duration': 0.3, 'activity': 'walk'},

                 {'start_time': "2024-02-13 07:50", 'duration': 0.3, 'activity': 'walk'},
                 {'start_time': "2024-02-13 09:50", 'duration': 0.3, 'activity': 'walk'},

                 {'start_time': "2024-02-14 14:15", 'duration': 0.3, 'activity': 'walk'},

                 {'start_time': "2024-02-15 07:50", 'duration': 0.2, 'activity': 'walk'},
                 {'start_time': "2024-02-15 13:40", 'duration': 0.5, 'activity': 'walk'},
                 {'start_time': "2024-02-15 17:40", 'duration': 0.3, 'activity': 'walk'},

                 {'start_time': "2024-02-17 17:00", 'duration': 0.67, 'activity': 'walk'},

                 {'start_time': "2024-02-18 17:30", 'duration': 0.5, 'activity': 'walk'},
                 {'start_time': "2024-02-18 22:00", 'duration': 0.5, 'activity': 'walk'}
                 ]
'''

class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        """
        This plot plots predicted trajectories from the measured values. It is recommended to use around 24 hours
        of test data for this plot to work optimally.

        #TODO: Predict trajectories when the models with different prediction horizons have the same config.
        """
        data_index = models_data[0]['y_true'].index
        start_date = data_index[0]
        end_date = data_index[-1]
        # print('start_date:', start_date)  
        # print("end_date:", end_date)  
        def on_hover(event):
            if event.inaxes == ax:
                for hover_line in lines:
                    contains, _ = hover_line.contains(event)
                    if contains:
                        hover_line.set_alpha(1.0)
                    else:
                        hover_line.set_alpha(0.2)
                fig.canvas.draw_idle()

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            unit = "mmol/L"

        unique_model_names = set()  # Initialize an empty set to store unique names
        unique_configs = set()
        unique_ph = set()
        trajectories = []  # Will be a set of trajectories for models with equal names and configs
        


        for model_entry in models_data:
            name = model_entry['name'].split(' ')[0]
            config = model_entry['config']
            ph = model_entry['prediction_horizon']
            unique_model_names.add(name)  # Add the name to the set
            unique_configs.add(config)
            unique_ph.add(ph)
        unique_model_names_list = list(unique_model_names)
        unique_configs_list = list(unique_configs)
        unique_ph_list = list(unique_ph)

        max_ph = max(unique_ph_list)
        max_target_index = max_ph // 5

        for model_name in unique_model_names_list:
            

            for config in unique_configs_list:
                filtered_entries = [entry for entry in models_data if
                                    entry['name'].split(' ')[0] == model_name and
                                    entry['config'] == config]

                if len(filtered_entries) != 0:
                    prediction_horizons = [0]
                    y_true = filtered_entries[0]['y_true'].dropna()
                    predictions = [y_true]

                    if not unit_config_manager.use_mgdl:
                        predictions = [[unit_config_manager.convert_value(val) for val in
                                       y_true]]

                    for entry in filtered_entries:
                        entry_prediction_horizon = entry['prediction_horizon']
                        start_index = entry_prediction_horizon // 5
                        prediction_horizons = prediction_horizons + [entry_prediction_horizon]
                        entry_predictions = entry['y_pred'][start_index:]

                        if unit_config_manager.use_mgdl:
                            predictions = predictions + [entry_predictions]
                        else:
                            predictions = predictions + [[unit_config_manager.convert_value(val) for val in
                                                          entry_predictions]]

                    # Sort lists by prediction horizons
                    pairs = list(zip(prediction_horizons, predictions))
                    sorted_pairs = sorted(pairs, key=lambda x: x[0])
                    sorted_prediction_horizons, sorted_predictions = zip(*sorted_pairs)

                    # Add trajectory data for reference value
                    trajectory_data = {
                        'prediction_horizons': list(sorted_prediction_horizons),
                        'predictions': list(sorted_predictions),  # A list of lists of predicted values
                        'y_true': y_true,
                        'model_name': model_name,
                        'config': config
                    }
                    trajectories.append(trajectory_data)

        for model_entry in models_data:
            name = model_entry['name'].split(' ')[0]
            config = model_entry['config']
            ph = model_entry['prediction_horizon']
            y_true = model_entry['y_true']
            total_time = len(y_true) * 5
            num_days = (total_time // (24 * 60)) + 1 


            # Get the period rmse for the model
            # rmse_at_specific_period_path = 'data/reports/period/rmse_at_specific_period.csv'
            # rmse_period = pd.read_csv(rmse_at_specific_period_path)
            # model_rmse_period = rmse_period[rmse_period['Model name'] == name]
            # period_rmse = model_rmse_period['Score'].iloc[0]
            # period_rmse = list(period_rmse.replace('[','').replace(']','').split(','))
            # counter = 0

            for day in range(num_days):
                start_index = day * 24 * 60 // 5
                end_index = min((day + 1) * 24 * 60 // 5, len(y_true))

                # Extract data for the current day
                y_true_day = y_true[start_index:end_index]
                fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size

                # Plot data
                ax.plot(y_true_day.index, y_true_day, label='Blood glucose measurements', color='black')

                # Plot each trajectory
                for trajectory_data in trajectories:
                    model_name = trajectory_data.get('model_name')
                    print("trajectory_data model_name: ", model_name)
                    y_prediction_lists = trajectory_data.get('predictions')
                    y_predictions = [y_prediction_lists[0][start_index:], y_prediction_lists[1][start_index:]]
                    # print('y_predictions:', y_predictions)
                    prediction_horizons = trajectory_data.get('prediction_horizons')

                    # Add predicted trajectories
                    for i in range(len(y_true_day)):
                        trajectory = []
                        t_values = []

                        # print("y_prediction_lists: ", y_prediction_lists)
                        # print("len(prediction_horizons): ", len(prediction_horizons))
                        for j in range(len(prediction_horizons)):
                            if i < len(y_predictions[j]):
                                t_values = t_values + [y_true_day.index[i] + datetime.timedelta(minutes=prediction_horizons[j])]
                                #print("y_prediction_lists[j][i]: ", y_prediction_lists[j][i])
                                trajectory = trajectory + [y_predictions[j][i]]
                            else:
                                break
                        #print('trajectory:', trajectory)
                        ax.plot(t_values, trajectory, linestyle='--', label=f'Predicted trajectory - {model_name}')

                ax.set_title(f'Predicted trajectories - Day {day + 1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Blood glucose')

                # format x-axis
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

                # format y-axis
                # ax.set_ylim(50, 350)
                ax.set_ylim(20, 280)
                ax.axhspan(70, 180, color='yellow', alpha=0.3)

                # print("y_true_day: ", y_true_day.index[0])
                # Mark the activity time
                mark_activity_time(activity_logs, y_true_day.index[0])

                # save plot
                save_dir = f"data/figures/" + model_name + '/'
                os.makedirs(save_dir, exist_ok=True)
                file_name = f"trajectories_day_{model_name}_{day + 1}.png"
                plt.savefig(os.path.join(save_dir, file_name))
                plt.close()

                print(f"Plot saved: {os.path.join(save_dir, file_name)}")



def mark_activity_time(periods, plot_start_date):
    plot_day = plot_start_date.date().strftime("%Y-%m-%d")
    for period in periods:
        # print("period['start_time'].split(' ')[0]: ", period['start_time'].split(' ')[0])
        # print("plot_start_date.date()", plot_start_date.date())
        
        if period['start_time'].split(' ')[0] == plot_day:
            start_time = datetime.datetime.strptime(period['start_time'], "%Y-%m-%d %H:%M")
            end_time = start_time + datetime.timedelta(hours=period['duration'])
            # print("period_rmse: ", period_rmse)
            # temp_rmse = period_rmse[counter]
            # counter += 1
            plt.axvline(x=start_time, color='r', linestyle='--')
            # plt.text(start_time, 230, '{}'.format(temp_rmse[:5]), verticalalignment='bottom')
            plt.axvline(x=end_time, color='r', linestyle='--')
    #return counter   

'''
        for trajectory_data in trajectories:
            fig, ax = plt.subplots()

            model_name = trajectory_data.get('model_name')
            y_prediction_lists = trajectory_data.get('predictions')
            y_true = trajectory_data.get('y_true')
            prediction_horizons = trajectory_data.get('prediction_horizons')

            total_time = len(y_true) * 5 + max_target_index * 5
            t = np.arange(0, total_time, 5)

            # convert 't' from minutes to datetime object, assuming t=0 is 00:00 of a day
            dates = [start_date + datetime.timedelta(minutes=int(ti)) for ti in t]

            # set x-axis major ticks format
            hours = mdates.HourLocator(interval = 2)  # Interval is 2 hours
            h_fmt = mdates.DateFormatter('%H:%M')

            ax.xaxis.set_major_locator(hours)
            ax.xaxis.set_major_formatter(h_fmt)

            # Set x-axis limits
            ax.set_xlim(dates[0], dates[-1])
            fig.autofmt_xdate()

            # Use correct unit
            if unit_config_manager.use_mgdl:
                ax.axhspan(70, 180, facecolor='blue', alpha=0.2)
            else:
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                ax.axhspan(unit_config_manager.convert_value(70), unit_config_manager.convert_value(180),
                           facecolor='blue', alpha=0.2)

            ax.set_title('Blood glucose predicted trajectories')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f'Blood glucose [{unit}]')
            ax.scatter(dates[:len(y_true)], y_true, label='Blood glucose measurements', color='black')

            lines = []
            # Add predicted trajectories
            for i in range(len(y_true)):
                trajectory = []
                t_values = []

                for j in range(len(prediction_horizons)):
                    if i < len(y_prediction_lists[j]):
                        t_values = t_values + [dates[i] + datetime.timedelta(minutes=prediction_horizons[j])]
                        trajectory = trajectory + [y_prediction_lists[j][i]]
                    else:
                        break

                line, = ax.plot(t_values, trajectory, linestyle='--')
                lines.append(line)

            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            ax.legend()
            plt.title(f'Predicted trajectories for {model_name}')
            file_path = "data/figures/"
            # TODO: Add config name
            file_name = f"trajectories_{trajectory_data.get('config')}_{model_name}.png"
            plt.savefig(file_path + file_name)
            plt.show()
''' 