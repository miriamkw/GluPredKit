import matplotlib.pyplot as plt
import ast
from .base_plot import BasePlot
from methcomp import parkes, clarke
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=30, type='parkes', *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        # Validate the type
        valid_types = {"parkes", "clarke"}
        if type not in valid_types:
            raise ValueError(f"Invalid type: {type}. Must be one of {valid_types}")

        for df in dfs:
            model_name = df['Model Name'][0]

            ph = int(df['prediction_horizon'][0])
            prediction_horizons = list(range(5, ph + 1, 5))

            if prediction_horizon:
                prediction_horizons = [prediction_horizon]

            y_true_values = []
            y_pred_values = []
            for prediction_horizon in prediction_horizons:
                y_true = df[f'target_{prediction_horizon}'][0]
                y_pred = df[f'y_pred_{prediction_horizon}'][0].replace("nan", "None")
                y_true = ast.literal_eval(y_true)
                y_pred = ast.literal_eval(y_pred)

                y_true_values += y_true
                y_pred_values += y_pred

            if unit_config_manager.get_unit() == 'mmol/L':
                y_true_values = [unit_config_manager.convert_value(val) for val in y_true_values]
                y_pred_values = [unit_config_manager.convert_value(val) for val in y_pred_values]
                units = "mmol"
            else:
                units = "mgdl"

            if len(prediction_horizons) >1:
                title = f"{type} error grid for {model_name} across all prediction horizons"
            else:
                title = f'{type} error grid for {model_name} at prediction horizon {prediction_horizon}'

            if type == 'parkes':
                parkes(1, y_true_values, y_pred_values, units=units,# x_label='', y_label=,
                       color_points="auto", grid=True, color_gridlabels='white',
                       percentage=False,
                       title=title)
            else:
                clarke(y_true_values, y_pred_values, units=units,  # x_label='', y_label=,
                       color_points="auto", grid=True, color_gridlabels='white',
                       percentage=False,
                       title=title)

            plt.show()

