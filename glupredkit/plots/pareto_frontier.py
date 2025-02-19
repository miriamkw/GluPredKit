import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base_plot import BasePlot


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=30, normalize_results=False, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        metrics = ['rmse', 'temporal_gain', 'g_mean']
        data = []
        plots = []
        names = []

        # Creates results df
        for df in dfs:
            model_name = df['Model Name'][0]
            row = {"Model Name": model_name}

            for metric_name in metrics:
                score = df[f'{metric_name}_{prediction_horizon}'][0]
                row[metric_name] = score
            data.append(row)

        results_df = pd.DataFrame(data)

        if normalize_results:
            prediction_horizon = float(prediction_horizon)
            results_df['temporal_gain'] = (prediction_horizon - results_df['temporal_gain']) / prediction_horizon
            results_df['g_mean'] = 1 - results_df['g_mean']
            normalized_df = results_df.copy()
            results_df[metrics] = normalized_df[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        def pareto_frontier(df):
            """
            Find the Pareto frontier from the DataFrame with metrics.
            This function returns the set of models that are not dominated by any other model.
            A model is dominated if another model has better (lower) performance across all metrics.
            """
            pareto_front = []

            # Change so that low value is better
            df['temporal_gain'] = -df['temporal_gain']
            df['g_mean'] = -df['g_mean']

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

            # Change back to original
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df['temporal_gain'] = -pareto_front_df['temporal_gain']
            pareto_front_df['g_mean'] = -pareto_front_df['g_mean']

            return pareto_front_df

        pareto_front_df = pareto_frontier(results_df.copy())

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

            # Highlight the optimal point
            ax.scatter(0.0, 1.0, float(prediction_horizon),
                       color='green', label='Optimal Point', s=150, marker='o')

            # Add annotations (model names) for each point
            for i, model_name in enumerate(df['Model Name']):
                ax.text(df['rmse'].iloc[i], df['g_mean'].iloc[i], df['temporal_gain'].iloc[i],
                        model_name, color='red', fontsize=10)

            ax.set_title('3D Plot: Precision vs Recall vs Temporal Accuracy - Pareto Frontier')
            ax.set_xlabel('rmse')
            ax.set_ylabel('g_mean')
            ax.set_zlabel('temporal_gain')
            ax.legend(loc='upper left')

        # Plot 3D visualization
        plot_3d(results_df, pareto_front_df)

        plot_name = f'pareto_frontier_ph_{prediction_horizon}'
        plots.append(plt.gcf())
        names.append(plot_name)

        if show_plot:
            plt.show()
        plt.close()

        return plots, names

