import glupredkit.helpers.cli as helpers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, show_plot=True, prediction_horizon=None, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        metrics = helpers.list_files_in_package('metrics')
        # Removing values and metrics that will not look reasonable in this format
        metrics = [val.split('.')[0] for val in metrics if not '__init__' in val and not 'base_metric' in val
                   and not 'error_grid' in val and not 'glycemia_detection' in val]
        data = []
        plots = []
        names = []

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
                    print(prediction_horizons)
                    results = []
                    for prediction_horizon in prediction_horizons:
                        score = df[f'{metric_name}_{prediction_horizon}'][0]
                        results += [score]
                    average_score = np.mean(results)
                    row[metric_name] = average_score

            data.append(row)

        results_df = pd.DataFrame(data)

        # Plotting the DataFrame as a table
        fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size as needed
        ax.axis("tight")  # Turn off the axes
        ax.axis("off")  # Turn off the axes completely

        # Add a title above the table
        title_text = f"Results for Prediction Horizon of {prediction_horizon} minutes"
        plt.text(0.0, 0.04, title_text, ha='center', va='center', fontsize=14, fontweight='bold', color='black')

        # Format numeric values to 2 decimals
        results_df.iloc[:, 1:] = results_df.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")

        # Create the table
        table = ax.table(
            cellText=results_df.values,  # Values of the table
            colLabels=results_df.columns,  # Column headers
            loc="center",  # Center the table
            cellLoc="center",  # Align cell text to center
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(results_df.columns))))  # Adjust column width

        # Make header row bold
        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0:  # Header row
                cell.set_text_props(weight="bold")  # Bold text
                cell.set_facecolor("lightgrey")

        # Add more spacing to cells
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(results_df.columns))))
        for cell in table.get_celld().values():
            cell.set_height(0.1)  # Adjust height for vertical padding
            cell.PAD = 0.05  # Increase cell padding

        plot_name = f'all_metrics_table_ph_{prediction_horizon}'
        plots.append(plt.gcf())
        names.append(plot_name)

        if show_plot:
            plt.show()
        plt.close()

        return plots, names

