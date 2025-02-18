import matplotlib.pyplot as plt
import numpy as np
import ast
from .base_plot import BasePlot
from methcomp import parkes, clarke
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, type='parkes', *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        # Validate the type
        valid_types = {"parkes", "clarke"}
        if type not in valid_types:
            raise ValueError(f"Invalid type: {type}. Must be one of {valid_types}")

        plots = []
        names = []

        for df in dfs:
            model_name = df['Model Name'][0]

            ph = int(df['prediction_horizon'][0])
            prediction_horizons = list(range(5, ph + 1, 5))

            fig, ax = plt.subplots(figsize=(6.5, 6))

            if prediction_horizon:
                prediction_horizons = [prediction_horizon]

            y_true_values = []
            y_pred_values = []
            for prediction_horizon in prediction_horizons:
                y_true = df[f'target_{prediction_horizon}'][0].replace("nan", "None")
                y_pred = df[f'y_pred_{prediction_horizon}'][0].replace("nan", "None")
                y_true = ast.literal_eval(y_true)
                y_pred = ast.literal_eval(y_pred)
                y_true = [np.nan if x is None else x for x in y_true]
                y_pred = [np.nan if x is None else x for x in y_pred]

                filtered_pairs = [(x, y) for x, y in zip(y_true, y_pred) if np.isfinite(x) and np.isfinite(y)]

                # Ensure filtered_pairs is not empty
                if not filtered_pairs:
                    print("No valid pairs of true and predicted values!")
                    return np.full((3, 3), np.nan)

                # Unpack the filtered pairs
                y_true, y_pred = map(list, zip(*filtered_pairs))

                y_true_values += y_true
                y_pred_values += y_pred

            if unit_config_manager.get_unit() == 'mmol/L':
                y_true_values = [unit_config_manager.convert_value(val) for val in y_true_values]
                y_pred_values = [unit_config_manager.convert_value(val) for val in y_pred_values]
                units = "mmol"
                x_label = 'Reference glucose concentration (mmol/L)'
                y_label = 'Predicted glucose concentration (mmol/L)'
                max_val = max(np.nanmax(y_true_values), np.nanmax(y_pred_values))
                max_val_rounded = np.round(max_val / 5) * 5
                custom_ticks = list(range(0, int(max_val_rounded) + 1, 5))
            else:
                units = "mgdl"
                x_label = 'Reference glucose concentration (mg/dL)'
                y_label = 'Predicted glucose concentration (mg/dL)'
                max_val = max(np.nanmax(y_true_values), np.nanmax(y_pred_values))
                max_val_rounded = np.ceil(max_val / 100) * 100
                custom_ticks = list(range(0, int(max_val_rounded) + 1, 100))

            xlim = max_val
            ylim = max_val

            if len(prediction_horizons) >1:
                title = f"{model_name} across all prediction horizons"
            else:
                title = f'{model_name} at prediction horizon {prediction_horizon}'

            if type == 'parkes':
                parkes(1, y_true_values, y_pred_values, units=units, x_label=x_label, y_label=y_label,
                       color_points="auto", grid=True, color_gridlabels='white', xlim=xlim, ylim=ylim,
                       percentage=False,
                       title=title, ax=ax)

            else:
                clarke(y_true_values, y_pred_values, units=units, x_label=x_label, y_label=y_label,
                       color_points="auto", grid=True, color_gridlabels='white',
                       percentage=False,
                       title=title)

            ax.set_xticks(custom_ticks)
            ax.set_yticks(custom_ticks)

            plot_name = f'{model_name}_{type}_error_grid_plot_ph_{prediction_horizon}'
            plots.append(plt.gcf())
            names.append(plot_name)

            if show_plot:
                plt.show()
            plt.close()

        return plots, names
