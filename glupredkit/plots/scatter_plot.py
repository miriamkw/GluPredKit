import matplotlib.pyplot as plt
import itertools
import os
import ast
import numpy as np
from datetime import datetime
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon, *args):
        """
        Plots the scatter plot for the given trained_models data.
        """
        # Define unique markers for scatter plot
        markers = itertools.cycle(('o', 's', '^', 'v', 'D', '<', '>'))

        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
            max_val = 400
        else:
            unit = "mmol/L"
            max_val = unit_config_manager.convert_value(400)

        plt.figure(figsize=(10, 8))

        for df in dfs:
            model_name = df['Model Name'][0]

            if f'target_{prediction_horizon}' not in df.columns:
                raise ValueError("The given prediction horizon is not within the total prediction horizon of the "
                                 "trained model. Please provide a valid prediction horizon.")

            y_true = df[f'target_{prediction_horizon}'][0]
            y_true = ast.literal_eval(y_true)

            y_pred = df[f'y_pred_{prediction_horizon}'][0]

            y_pred = y_pred.replace("nan", "None")
            y_pred = ast.literal_eval(y_pred)
            y_pred = [np.nan if val is None else val for val in y_pred]

            if not unit_config_manager.use_mgdl:
                y_pred = [unit_config_manager.convert_value(val) for val in y_pred]
                y_true = [unit_config_manager.convert_value(val) for val in y_true]

            marker = next(markers)

            plt.scatter(y_true, y_pred, label=model_name, marker=marker, alpha=0.5)

        # Plotting the line x=y
        plt.plot([0, max_val], [0, max_val], 'k-')

        plt.xlabel(f"True Blood Glucose [{unit}]")
        plt.ylabel(f"Predicted Blood Glucose [{unit}]")
        plt.title(f"Scatter Plot of Prediction Accuracy {prediction_horizon} Minutes Prediction Horizon")
        plt.legend(loc='upper left')

        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)

        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
        safe_timestamp = safe_timestamp.replace('.', '_')
        file_name = f'scatter_plot_{safe_timestamp}.png'
        plt.savefig(file_path + file_name)
        plt.show()
