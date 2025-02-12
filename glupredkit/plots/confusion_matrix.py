import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import seaborn as sns
from datetime import datetime
from .base_plot import BasePlot


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=30, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        classes = ['Hypo', 'Target', 'Hyper']

        plots = []
        names = []
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
            plt.figure(figsize=(7, 5.8))
            sns.heatmap(average_matrix, annot=True, cmap=plt.cm.Blues, fmt='.1%', xticklabels=classes,
                        yticklabels=classes)
            if len(prediction_horizons) > 1:
                plt.title(f'Total Over all PHs for {model_name}')
            else:
                plt.title(f'PH {prediction_horizon} for {model_name}')
            plt.xlabel('True label')
            plt.ylabel('Predicted label')

            file_path = "data/figures/"
            os.makedirs(file_path, exist_ok=True)

            plot_name = f'{model_name}_confusion_matrix_ph_{prediction_horizon}'
            plots.append(plt.gcf())
            names.append(plot_name)
            plt.close()

        return plots, names

