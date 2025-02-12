import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=None, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        """
        # TODO: Add the colour coding, that can be turned on/off on input

        metrics = ['rmse', 'temporal_gain', 'g_mean']#, 'me']
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

        print(results_df)
        print(prediction_horizon)

        # Plotting the DataFrame as a table
        fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size as needed
        ax.axis("tight")  # Turn off the axes
        ax.axis("off")  # Turn off the axes completely

        # Add a title above the table
        title_text = f"Results for Prediction Horizon of {prediction_horizon} minutes"
        plt.text(0.0, 0.04, title_text, ha='center', va='center', fontsize=14, fontweight='bold', color='black')

        scaled_rmse = scale_errors(results_df['rmse'], use_mg_dl=unit_config_manager.use_mgdl)
        prediction_horizon = float(prediction_horizon)
        scaled_tg = np.array((prediction_horizon - results_df['temporal_gain']) / prediction_horizon)
        scaled_g_mean = np.array(1 - results_df['g_mean'])
        #scaled_me = scale_errors(results_df['me'], use_mg_dl=unit_config_manager.use_mgdl)

        # Add CGPM
        results_df['CGPM'] = scaled_rmse + scaled_tg + scaled_g_mean # + scaled_me

        # Format numeric values to 2 decimals
        results_df.iloc[:, 1:] = results_df.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")

        # Add parenthesis for scaled version
        results_df['rmse'] = add_scaled_value_to_result(results_df['rmse'], scaled_rmse)
        results_df['temporal_gain'] = add_scaled_value_to_result(results_df['temporal_gain'], scaled_tg)
        results_df['g_mean'] = add_scaled_value_to_result(results_df['g_mean'], scaled_g_mean)
        #results_df['me'] = add_scaled_value_to_result(results_df['me'], scaled_me)

        # Map values to prettier strings
        results_df.rename(columns={'rmse': 'RMSE', 'temporal_gain': 'Temporal Gain', 'g_mean': 'G-Mean',
                                   'me': 'Mean Error'}, inplace=True)

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
            cell.PAD = 0.1  # Increase cell padding

        plot_name = f'cgpm_table_ph_{prediction_horizon}'
        plots.append(plt.gcf())
        names.append(plot_name)
        plt.close()

        return plots, names

def scale_errors(metric_results, use_mg_dl=False):
    lower_bound = 1
    if use_mg_dl:
        lower_bound = 18
    max_abs = max(np.max(np.abs(metric_results)), lower_bound)
    return np.array([np.abs(val) / max_abs for val in metric_results])


def add_scaled_value_to_result(results, scaled_results):
    return results + ' (' + [f"{value:.2f}" for value in scaled_results] + ')'

