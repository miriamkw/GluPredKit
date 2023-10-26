from .base_plot import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        """
        This plot plots predicted trajectories from the measured values. It is recommended to use around 24 hours
        of test data for this plot to work optimally.

        #TODO: Predict trajectories when the models with different prediction horizons have the same config.
        """

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
        max_target_index = max_ph // 5

        trajectories = []  # Will be a set of trajectories for models with equal names and configs

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
                        'config': config,
                    }
                    trajectories.append(trajectory_data)

        for trajectory_data in trajectories:
            fig, ax = plt.subplots()

            model_name = trajectory_data.get('model_name')
            y_prediction_lists = trajectory_data.get('predictions')
            y_true = trajectory_data.get('y_true')
            prediction_horizons = trajectory_data.get('prediction_horizons')

            total_time = len(y_true) * 5 + max_target_index * 5
            t = np.arange(0, total_time, 5)

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
            ax.scatter(t[:len(y_true)], y_true, label='Blood glucose measurements', color='black')

            lines = []
            # Add predicted trajectories
            for i in range(len(y_true)):
                trajectory = []
                t_values = []

                for j in range(len(prediction_horizons)):
                    if i < len(y_prediction_lists[j]):
                        t_values = t_values + [i*5 + prediction_horizons[j]]
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
