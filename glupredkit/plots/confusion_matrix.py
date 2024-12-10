import matplotlib.pyplot as plt
import itertools
import os
import ast
import numpy as np
import seaborn as sns

from datetime import datetime
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=30, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        classes = ['Hypo', 'Target', 'Hyper']

        for df in dfs:
            model_name = df['Model Name'][0]

            ph = int(df['prediction_horizon'][0])
            prediction_horizons = list(range(5, ph + 1, 5))

            if prediction_horizon:
                prediction_horizons = [prediction_horizon]

            results = []
            for prediction_horizon in prediction_horizons:
                percentages = df[f'glycemia_detection_{prediction_horizon}'][0]
                percentages = ast.literal_eval(percentages)
                results += [percentages]

            matrix_array = np.array(results)
            average_matrix = np.mean(matrix_array, axis=0)

            plt.figure(figsize=(8, 6))
            sns.heatmap(average_matrix, annot=True, cmap=plt.cm.Blues, fmt='.2%', xticklabels=classes, yticklabels=classes)
            if len(prediction_horizons) > 1:
                plt.title(f'Total Over all PHs for {model_name}')
            else:
                plt.title(f'PH {prediction_horizon} for {model_name}')
            plt.xlabel('True label')
            plt.ylabel('Predicted label')

            file_path = "data/figures/"
            os.makedirs(file_path, exist_ok=True)

            timestamp = datetime.now().isoformat()
            safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
            safe_timestamp = safe_timestamp.replace('.', '_')

            file_name = f'confusion_matrix_{safe_timestamp}_{model_name}.png'
            plt.savefig(file_path + file_name)
            plt.show()

