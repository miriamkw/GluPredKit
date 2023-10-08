import matplotlib.pyplot as plt
import itertools
import os
from .base_plot import BasePlot
from ..config_manager import config_manager


class Plot(BasePlot):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def __call__(self, models_data, y_true):
        """
        Plots the scatter plot for the given trained_models data.

        models_data: A list of dictionaries containing the model name, y_true, and y_pred.
                    Example: [{'name': 'model1', 'y_pred': [...]}, ...]
        """
        # Define unique markers for scatter plot
        markers = itertools.cycle(('o', 's', '^', 'v', 'D', '<', '>'))

        if config_manager.use_mgdl:
            unit = "mg/dL"
        else:
            y_true = [config_manager.convert_value(val) for val in y_true]
            unit = "mmol/L"

        plt.figure(figsize=(10, 8))

        for model_data in models_data:
            model_name = model_data.get('name')

            if config_manager.use_mgdl:
                y_pred = model_data.get('y_pred')
            else:
                y_pred = [config_manager.convert_value(val) for val in model_data.get('y_pred')]

            marker = next(markers)

            plt.scatter(y_true, y_pred, label=model_name, marker=marker)

        # Plotting the line x=y
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k-')

        plt.xlabel(f"True Blood Glucose [{unit}]")
        plt.ylabel(f"Predicted Blood Glucose [{unit}]")
        plt.title(f"Accuracy of {self.prediction_horizon}-minutes ahead predictions")
        plt.legend(loc='upper left')

        file_path = "results/figures/"
        os.makedirs(file_path, exist_ok=True)

        file_name = f'scatter_plot_ph-{self.prediction_horizon}.png'
        plt.savefig(file_path + file_name)
        plt.show()