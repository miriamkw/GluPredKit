import matplotlib.pyplot as plt
import itertools
import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import glupredkit.helpers.cli as helpers

from datetime import datetime
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=120, *args):
        """
        Plots the confusion matrix for the given trained_models data.

        mode 0: normal metrics
        mode 1: using equally weighted scores
        mode 2: using cost function (severity blood glucose and gradient)
        """
        # TODO: Implement overall vs prediction horizon
        metrics = ['rmse', 'temporal_gain', 'g_mean']
        data = []
        prediction_horizons = []

        # Creates results df
        for df in dfs:
            model_name = df['Model Name'][0]
            row = {"Model Name": model_name}

            for metric_name in metrics:
                # If prediction horizon is defined, add metric at prediction horizon.
                # If not, add total across all prediction horizons
                if prediction_horizon:
                    score = df[f'{metric_name}_{prediction_horizon}'][0]
                    row[metric_name] = score
                else:
                    ph = int(df['prediction_horizon'][0])
                    prediction_horizons = list(range(5, ph + 1, 5))
                    results = []
                    for prediction_horizon in prediction_horizons:
                        score = df[f'{metric_name}_{prediction_horizon}'][0]
                        results += [score]
                    average_score = np.mean(results)
                    row[metric_name] = average_score

            data.append(row)

        results_df = pd.DataFrame(data)
        results_df['temporal_gain'] = (prediction_horizon - results_df['temporal_gain']) / prediction_horizon
        results_df['g_mean'] = 1 - results_df['g_mean']
        normalized_df = results_df.copy()
        normalized_df[metrics] = normalized_df[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        print("NORM", normalized_df)

        def pareto_frontier(df):
            """
            Find the Pareto frontier from the DataFrame with metrics.
            This function returns the set of models that are not dominated by any other model.
            A model is dominated if another model has better (lower) performance across all metrics.
            """
            pareto_front = []

            # Iterate through each model (row)
            for i, model in df.iterrows():
                is_dominated = False

                # Compare with every other model (row)
                for j, other_model in df.iterrows():
                    if (other_model < model).all():  # if all metrics of other_model are better (lower)
                        is_dominated = True
                        break  # Stop comparing this model (i) with further models, it's dominated

                if not is_dominated:
                    pareto_front.append(model)  # This model is not dominated, so add to Pareto front

            return pd.DataFrame(pareto_front)

        print("results", results_df)

        pareto_front_df = pareto_frontier(results_df)
        print("Pareto Frontier:")
        print(pareto_front_df)

        def plot_2d(df, pareto_front_df):
            """
            Plot the Precision vs Recall and highlight the Pareto frontier
            """
            plt.figure(figsize=(10, 6))

            # Scatter plot for all models
            plt.scatter(df['rmse'], df['g_mean'], color='blue', label='All Models', s=100, alpha=0.7)

            # Highlight Pareto frontier
            plt.scatter(pareto_front_df['rmse'], pareto_front_df['g_mean'], color='red', label='Pareto Frontier',
                        s=100, marker='x')

            plt.title('Precision vs Recall - Pareto Frontier')
            plt.xlabel('RMSE')
            plt.ylabel('G-Mean')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.show()

        # Plot 2D
        #plot_2d(results_df, pareto_front_df)

        def plot_3d(df, pareto_front_df):
            """
            Plot Precision vs Recall vs Temporal Accuracy and highlight the Pareto frontier
            """
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for all models
            ax.scatter(df['rmse'], df['g_mean'], df['temporal_gain'], color='blue', label='All Models', s=100,
                       alpha=0.7)

            # Highlight Pareto frontier
            ax.scatter(pareto_front_df['rmse'], pareto_front_df['g_mean'], pareto_front_df['temporal_gain'],
                       color='red', label='Pareto Frontier', s=100, marker='x')

            # Add annotations (model names) for each point
            for i, model_name in enumerate(df['Model Name']):
                ax.text(df['rmse'].iloc[i], df['g_mean'].iloc[i], df['temporal_gain'].iloc[i],
                        model_name, color='red', fontsize=10)

            ax.set_title('3D Plot: Precision vs Recall vs Temporal Accuracy - Pareto Frontier')
            ax.set_xlabel('rmse')
            ax.set_ylabel('g_mean')
            ax.set_zlabel('temporal_gain')
            ax.legend(loc='upper left')
            plt.show()

        # Plot 3D visualization
        plot_3d(results_df, pareto_front_df)


