from .base_plot import BasePlot
import ast
import random
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from glupredkit.helpers.unit_config_manager import unit_config_manager
import glupredkit.helpers.cli as helpers
from datetime import datetime

class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, start_index=12*12*3, n_samples=12*12, prediction_horizon=120, *args):
        # TODO: Add these input options to the cli
        # TODO: Fix the plot when using max samples in evaluate model
        """
        This plot plots predicted trajectories from the measured values. A random subsample of around 24 hours will
        be plotted.
        """
        # Choose the data from the first subject in dataset

        for df in dfs:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(14, 8))

            model_name = df['Model Name'][0]

            dates_string_list = df['test_input_date'][0]
            datetime_strings = re.findall(r"'(.*?)'", dates_string_list)
            timestamp_list = [pd.Timestamp(ts) for ts in datetime_strings]

            # TODO: Only use the types that are present in the input
            # TODO: We only use the first 2000 elements to use data from only one subject...
            # Create a DataFrame
            model_df = pd.DataFrame({
                'date': timestamp_list[12000:14000],
                'CGM': get_list_from_string(df, 'test_input_CGM')[12000:14000],
                'y_pred': get_list_from_string(df, f'y_pred_{prediction_horizon}')[12000:14000],
            })
            model_df.set_index('date', inplace=True)
            full_time_range = pd.date_range(start=model_df.index.min(), end=model_df.index.max(), freq='5T')
            # TODO: We need to fix these plots because they are not designed to handle different subjects in one df
            model_df = model_df.reindex(full_time_range)

            if n_samples > model_df.shape[0]:
                n_samples = model_df.shape[0]

            if start_index is not None:
                print("start index true")
                if start_index > model_df.shape[0] - n_samples:
                    print(f"Start index too high. Should be below {model_df.shape[0] - n_samples}. Setting start index to 0...")
                    start_index = 0
            elif model_df.shape[0] > n_samples:
                start_index = random.randint(0, model_df.shape[0] - n_samples)
            else:
                start_index = 0

            # Filter model df based on start index and num samples
            model_df = model_df.iloc[start_index:start_index + n_samples]

            total_time = n_samples * 5
            t = np.arange(0, total_time, 5)

            # TODO: Make sure that both units works for this plot
            y_true = model_df['CGM'].tolist()
            y_pred = model_df['y_pred'].tolist()
            y_pred_original = y_pred
            y_true_original = y_true
            if unit_config_manager.use_mgdl:
                ax1.axhspan(70, 180, facecolor='blue', alpha=0.2)
            else:
                y_true = [unit_config_manager.convert_value(val) for val in y_true]
                y_pred = [unit_config_manager.convert_value(val) for val in y_pred]
                ax1.axhspan(unit_config_manager.convert_value(70), unit_config_manager.convert_value(180),
                           facecolor='blue', alpha=0.2)

            ax1.set_title(f'Predictions at PH {prediction_horizon} minutes for {model_name}', fontsize=16)
            ax1.set_ylabel(f'Blood glucose [{unit_config_manager.get_unit()}]')

            ax1.scatter(t, y_true, label='Measurements', color='black')
            ax1.set_xlim(0, n_samples * 5)

            # Plot line for predicted values
            t_pred = [val + prediction_horizon for val in t]
            ax1.plot(t_pred, y_pred[:len(t_pred)], color='green', linestyle='--', label=f'{model_name} Predictions')

            # Second plot, metrics weights
            ax2.set_title('Metric Weights', fontsize=14)
            # TODO: Create a list of values for each metric!
            # TODO: Create also a horizontal line for the average final score (?)
            aligned_y_true = y_true_original[prediction_horizon // 5:]
            aligned_y_pred = y_pred_original[:-prediction_horizon // 5]
            squared_errors = [(yt - yp)**2 for yt, yp in zip(aligned_y_true, aligned_y_pred)]

            parkes_linear_metric_module = helpers.get_metric_module('parkes_linear')
            parkes_linear_metric = parkes_linear_metric_module.Metric()
            parkes_errors = [parkes_linear_metric([yt], [yp]) for yt, yp in zip(aligned_y_true, aligned_y_pred)]

            seg_module = helpers.get_metric_module('seg')
            seg_metric = seg_module.Metric()
            seg_errors = [seg_metric([yt], [yp]) for yt, yp in zip(aligned_y_true, aligned_y_pred)]

            def normalize(data):
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            ax2.plot(t[prediction_horizon // 5:], normalize(squared_errors), label='Squared Errors')
            #ax2.step(t[prediction_horizon // 5:], normalize(parkes_errors), label='Parkes Linear Errors')
            ax2.plot(t[prediction_horizon // 5:], normalize(seg_errors), label='Surveillance Error Grid Scores')
            ax2.set_xlim(0, n_samples * 5)

            # Third plot, adding cost function
            ax3.set_title('Metric Weights with Cost Function', fontsize=14)
            ax3.set_xlabel('Time (minutes)', fontsize=14)

            prev_y_true = y_true_original[prediction_horizon // 5 - 1:-1]
            delta_bgs = [val - prev_val for (val, prev_val) in zip(aligned_y_true, prev_y_true)]
            severity_weights = [slope_cost(bg, delta_bg) + zone_cost(bg) + 1 for bg, delta_bg in zip(aligned_y_true, delta_bgs)]

            squared_errors_with_cost = [error * weight for error, weight in zip(squared_errors, severity_weights)]
            parkes_errors_with_cost = [error * weight for error, weight in zip(parkes_errors, severity_weights)]
            seg_errors_with_cost = [error * weight for error, weight in zip(seg_errors, severity_weights)]

            ax3.plot(t[prediction_horizon // 5:], normalize(squared_errors_with_cost))
            #ax3.step(t[prediction_horizon // 5:], normalize(parkes_errors_with_cost))
            ax3.plot(t[prediction_horizon // 5:], normalize(seg_errors_with_cost))
            ax3.set_xlim(0, n_samples * 5)

            # Remove ticks on first and second plot
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])

            # Collect all legends into one frame
            fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))  # Adjust position as needed
            file_path = "data/figures/"

            timestamp = datetime.now().isoformat()
            safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
            safe_timestamp = safe_timestamp.replace('.', '_')
            file_name = f"trajectories_{model_name}_{safe_timestamp}.png"
            plt.savefig(file_path + file_name)
            plt.show()


def get_list_from_string(df, col):
    string_values = df[col][0]
    string_values = string_values.replace("nan", "None")
    list_values = ast.literal_eval(string_values)
    list_values = [np.nan if x is None else x for x in list_values]
    return list_values



def slope_cost(bg, delta_bg):
    k = 18.0182
    # This function assumes mmol/L
    bg = bg / k
    delta_bg = delta_bg / k

    a = bg
    b = 15 - bg
    if b < 0:
        b = 0
    if a > 15:
        a = 15

    cost = (np.sign(delta_bg) + 1) / 2 * a * (delta_bg ** 2) - 2 * (np.sign(delta_bg) - 1) / 2 * b * (delta_bg ** 2)
    return cost


def zone_cost(bg, target=105):
    if bg < 1:
        bg = 1
    if bg > 600:
        bg = 600

    # This function already assumes BG in mg / dL
    constant = 32.9170208165394
    risk = constant * (np.log(bg) - np.log(target)) ** 2

    # TODO: We must tune this!

    return risk




