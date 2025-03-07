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

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, *args):
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
                if isinstance(percentages, str):
                    percentages = percentages.replace("nan", "None")
                    percentages = ast.literal_eval(percentages)
                percentages = [
                    [np.nan if x is None else x for x in sublist] for sublist in percentages
                ]
                results += [percentages]

            matrix_array = np.array(results)
            average_matrix = np.nanmean(matrix_array, axis=0)
            plt.figure(figsize=(7, 5.8))
            sns.heatmap(average_matrix, annot=True, cmap=plt.cm.Blues, fmt='.1%', xticklabels=classes,
                        yticklabels=classes)
            if len(prediction_horizons) > 1:
                plt.title(f'Total Over all PHs for {model_name}')
            else:
                plt.title(f'PH {prediction_horizon} for {model_name}')
            plt.xlabel('True label')
            plt.ylabel('Predicted label')

            plot_name = f'{model_name}_confusion_matrix_ph_{prediction_horizon}'
            plots.append(plt.gcf())
            names.append(plot_name)

            if show_plot:
                plt.show()

            plt.close()

        return plots, names

