import matplotlib.pyplot as plt
import ast
from .base_plot import BasePlot
from methcomp import parkeszones, clarkezones
from collections import Counter
import pandas as pd

class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=30, type='parkes', *args):
        # Validate the type
        valid_types = {"parkes", "clarke"}
        if type not in valid_types:
            raise ValueError(f"Invalid type: {type}. Must be one of {valid_types}")

        data = []
        plots = []
        names = []
        for df in dfs:
            model_name = df['Model Name'][0]
            row = {"Model Name": model_name}

            ph = int(df['prediction_horizon'][0])
            prediction_horizons = list(range(5, ph + 1, 5))

            if prediction_horizon:
                prediction_horizons = [prediction_horizon]

            y_true_values = []
            y_pred_values = []
            for prediction_horizon in prediction_horizons:
                y_true = df[f'target_{prediction_horizon}'][0]
                y_pred = df[f'y_pred_{prediction_horizon}'][0].replace("nan", "None")
                y_true = ast.literal_eval(y_true)
                y_pred = ast.literal_eval(y_pred)

                y_true_values += y_true
                y_pred_values += y_pred

            if type == 'parkes':
                zones = parkeszones(1, y_true_values, y_pred_values, units="mgdl", numeric=False)

            else:
                zones = clarkezones(y_true_values, y_pred_values, units="mgdl", numeric=False)

            counter = Counter(zones)
            n_total = len(zones)

            row['A [%]'] = counter['A'] / n_total * 100
            row['B [%]'] = counter['B'] / n_total * 100
            row['C [%]'] = counter['C'] / n_total * 100
            row['D [%]'] = counter['D'] / n_total * 100
            row['E [%]'] = counter['E'] / n_total * 100
            row['Weighted Average'] = (counter['B'] + 2*counter['C'] + 3*counter['D'] + 4*counter['E']) / n_total

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

        plot_name = f'{type}_error_grid_table_ph_{prediction_horizon}'
        plots.append(plt.gcf())
        names.append(plot_name)
        plt.close()

        return plots, names

