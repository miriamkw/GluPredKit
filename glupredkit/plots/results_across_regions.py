import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=30, metric='mean_error', *args):

        # Whether to plot RMSE or Mean Error
        use_rmse = False
        if metric == 'rmse':
            use_rmse = True

        for df in dfs:
            model_name = df['Model Name'][0]

            # Get results:
            y_true = df[f'target_{prediction_horizon}'][0]  # PH doesnt really matter that much
            y_true = ast.literal_eval(y_true)
            y_true = np.array(y_true)
            y_pred = df[f'y_pred_{prediction_horizon}'][0]
            y_pred = ast.literal_eval(y_pred)
            y_pred = np.array(y_pred)

            # Define bins based on y_true values
            bin_edges = [0, 70, 180, np.inf]  # Bins: <70, 70-180, >180
            bin_labels = ['<70', '70-180', '>180']

            error_values = []

            # Calculate RMSE for each bin
            for i in range(len(bin_edges) - 1):
                lower_edge = bin_edges[i]
                upper_edge = bin_edges[i + 1]
                bin_mask = (y_true >= lower_edge) & (y_true < upper_edge)
                if use_rmse:
                    error = calculate_rmse(y_true[bin_mask], y_pred[bin_mask])
                else:
                    error = calculate_mean_error(y_true[bin_mask], y_pred[bin_mask])
                error_values.append(error)

            # Plot
            plt.figure(figsize=(10, 6))

            plt.bar(bin_labels, error_values, color='skyblue')
            plt.xlabel('Bin Range')
            if use_rmse:
                plt.ylabel('RMSE')
                plt.title(f'{model_name} RMSE by Bins of y_true')
            else:
                plt.ylabel('Mean Error')
                plt.title(f'{model_name} Mean Error by Bins of y_true')

            plt.show()


def calculate_rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def calculate_mean_error(y_true, y_pred):
    """Calculate Mean Error."""
    return np.mean(y_pred - y_true)





