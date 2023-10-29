import matplotlib.pyplot as plt
import itertools
import os
from datetime import datetime
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        """
        Plots the scatter plot for the given trained_models data.

        models_data: A list of dictionaries containing the model name, y_true, and y_pred.
                    Example: [{'name': 'model1', 'y_pred': [...]}, ...]
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

        for model_data in models_data:
            model_name = model_data.get('name')

            if unit_config_manager.use_mgdl:
                y_pred = model_data.get('y_pred')
                y_true = model_data.get('y_true')
            else:
                y_pred = [unit_config_manager.convert_value(val) for val in model_data.get('y_pred')]
                y_true = [unit_config_manager.convert_value(val) for val in model_data.get('y_true')]

            marker = next(markers)

            plt.scatter(y_true, y_pred, label=model_name, marker=marker, alpha=0.5)

        # Plotting the line x=y
        plt.plot([0, max_val], [0, max_val], 'k-')

        plt.xlabel(f"True Blood Glucose [{unit}]")
        plt.ylabel(f"Predicted Blood Glucose [{unit}]")
        plt.title(f"Scatter Plot of Prediction Accuracy")
        plt.legend(loc='upper left')

        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)

        file_name = f'scatter_plot_{datetime.now()}.png'
        plt.savefig(file_path + file_name)
        plt.show()
