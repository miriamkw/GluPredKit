import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime, timedelta
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        """
        Plots the scatter plot for the given trained_models data. This plot automatically use the last data for
        plotting.

        models_data: A list of dictionaries containing the model name, y_true, y_pred, prediction horizon and the name
        of the configuration file.
        """
        history_length = 18  # Number of samples of measured CGM values
        sample_rate = 5  # CGM values are sampled every 5 minutes

        t = np.arange(-history_length * sample_rate + sample_rate, 0 + sample_rate, sample_rate)

        unique_model_names = set()  # Initialize an empty set to store unique names
        unique_configs = set()
        unique_ph = set()

        real_time = False
        for model_entry in models_data:
            name = model_entry['name'].split(' ')[0]
            real_time = model_entry['real_time']
            config = model_entry['config']
            ph = model_entry['prediction_horizon']
            unique_model_names.add(name)  # Add the name to the set
            unique_configs.add(config)
            unique_ph.add(ph)

        unique_model_names_list = list(unique_model_names)
        unique_configs_list = list(unique_configs)
        unique_ph_list = list(unique_ph)

        max_ph = max(unique_ph_list)
        max_target_index = max_ph // sample_rate

        trajectories = []  # Will be a set of trajectories for models with equal names and configs

        for model_name in unique_model_names_list:
            for config in unique_configs_list:
                filtered_entries = [entry for entry in models_data if
                                    entry['name'].split(' ')[0] == model_name and
                                    entry['config'] == config]

                if len(filtered_entries) != 0:
                    prediction_horizons = [0]

                    if real_time:
                        predictions = [filtered_entries[0]['y_true'].dropna().iloc[-1]]
                    else:
                        predictions = [filtered_entries[0]['y_true'].iloc[-1 - max_target_index]]

                    for entry in filtered_entries:
                        ph = entry['prediction_horizon']

                        if real_time:
                            prediction = [entry['y_pred'][-1]]
                        else:
                            target_index = ph // sample_rate
                            prediction = [entry['y_pred'][-target_index]]

                        prediction_horizons = prediction_horizons + [ph]
                        predictions = predictions + prediction

                    if not unit_config_manager.use_mgdl:
                        predictions = [unit_config_manager.convert_value(val) for val in predictions]

                    # Sort lists by prediction horizons
                    pairs = list(zip(prediction_horizons, predictions))
                    sorted_pairs = sorted(pairs, key=lambda x: x[0])
                    sorted_prediction_horizons, sorted_predictions = zip(*sorted_pairs)

                    # Add trajectory data for reference value
                    trajectory_data = {
                        'prediction_horizons': list(sorted_prediction_horizons),
                        'predictions': list(sorted_predictions),
                        'model_name': model_name,
                    }
                    trajectories.append(trajectory_data)

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            unit = "mmol/L"

        # TODO: Add potential carbs and insulin input (DRAW IT AS WELL!)

        plt.figure(figsize=(10, 8))

        for trajectory in trajectories:
            plt.plot(trajectory.get('prediction_horizons'), trajectory.get('predictions'),
                     label=f"Predictions for {trajectory.get('model_name')}")

        # This iteration will break after the first iteration, using a random sample of model data
        for model_data in models_data:
            y_true = model_data.get('y_true')
            prediction_horizon = model_data.get('prediction_horizon')

            if real_time:
                last_y_true_values = y_true.dropna()[-history_length:]
            else:
                last_y_true_values = y_true[-history_length - max_target_index:-max_target_index]

            # Get the datetime where t=0. The index/date of y_true is the date of the predicted value
            time_of_prediction = last_y_true_values.index[-1] + timedelta(minutes=prediction_horizon)

            # User correct unit. Add a text label above t=0
            if unit_config_manager.use_mgdl:
                plt.text(0, 5, f'{time_of_prediction}', fontsize=12, ha='center')
                if model_data.get('carbs'):
                    plt.text(0, 20, f"{model_data.get('carbs')}g of carbohydrates", fontsize=12, ha='center')
                if model_data.get('insulin'):
                    plt.text(0, 30, f"{model_data.get('insulin')}U of insulin", fontsize=12, ha='center')
            else:
                plt.text(0, unit_config_manager.convert_value(5), f'{time_of_prediction}', fontsize=12, ha='center')
                last_y_true_values = [unit_config_manager.convert_value(val) for val in last_y_true_values]
                if model_data.get('carbs'):
                    plt.text(0, unit_config_manager.convert_value(20), f"{model_data.get('carbs')}g of carbohydrates",
                             fontsize=12, ha='center')
                if model_data.get('insulin'):
                    plt.text(0, unit_config_manager.convert_value(30), f"{model_data.get('insulin')}U of insulin",
                             fontsize=12, ha='center')

            plt.scatter(t, last_y_true_values, label=f'Blood glucose measurements', color='black')
            break

        if unit_config_manager.use_mgdl:
            plt.axhspan(70, 180, facecolor='blue', alpha=0.2)
        else:
            plt.axhspan(unit_config_manager.convert_value(70), unit_config_manager.convert_value(180),
                        facecolor='blue', alpha=0.2)

        # Get current y-axis limits
        y_min, y_max = plt.ylim()

        # Set new y-axis limits to include 0
        plt.ylim(min(0, int(y_min)), y_max)

        plt.xlabel(f"Time [minutes]")
        plt.ylabel(f"Blood glucose [{unit}]")
        plt.title(f"Single Prediction Plot")
        plt.legend(loc='upper left')

        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)

        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        file_name = f'one_prediction_{safe_timestamp}.png'
        plt.savefig(file_path + file_name)
        plt.show()
