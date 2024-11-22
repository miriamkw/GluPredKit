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

    def __call__(self, dfs, prediction_horizon=30, zone_weighting_mode=0, *args):
        """
        Plots the confusion matrix for the given trained_models data.

        mode 0: normal metrics
        mode 1: using equally weighted scores
        mode 2: using cost function (severity blood glucose and gradient)
        """
        # TODO: Implement overall vs prediction horizon
        # TODO: Implement metrics as an input value
        #metrics = ['rmse', 'parkes_linear', 'temporal_gain', 'mcc', 'f1']
        #metrics = ['mcc', 'f1', 'g_mean']
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
                if zone_weighting_mode == 0:
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
                elif zone_weighting_mode == 1:
                    metric_module = helpers.get_metric_module(metric_name)
                    metric = metric_module.Metric()

                    if prediction_horizon:
                        scores = get_equally_weighted_scores_for_metric(df, prediction_horizon, metric)
                        if scores:
                            average_score = sum(scores) / len(scores)
                        else:
                            average_score = None  # Handle the case where no valid scores are present
                        row[metric_name] = average_score
                    else:
                        ph = int(df['prediction_horizon'][0])
                        prediction_horizons = list(range(5, ph + 1, 5))
                        results = []
                        for prediction_horizon in prediction_horizons:
                            scores = get_equally_weighted_scores_for_metric(df, prediction_horizon, metric)
                            if scores:
                                average_score = sum(scores) / len(scores)
                            else:
                                average_score = None  # Handle the case where no valid scores are present
                            results += [average_score]
                        average_score = np.mean(results)
                        row[metric_name] = average_score

                elif zone_weighting_mode == 2:
                    metric_module = helpers.get_metric_module(metric_name)
                    metric = metric_module.Metric()

                    if prediction_horizon:
                        scores = get_scores_for_metric(df, prediction_horizon, metric)
                        average_score = sum(scores) / len(scores)
                        row[metric_name] = average_score
                    # TODO: Add total across PHs option

            data.append(row)

        results_df = pd.DataFrame(data)

        print("results", results_df)
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
            cell.PAD = 0.01  # Increase cell padding

        plt.show()


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


def get_equally_weighted_scores_for_metric(df, ph, metric):
    y_true = df[f'target_{ph}'][0]
    y_pred = df[f'y_pred_{ph}'][0].replace("nan", "None")
    y_true = ast.literal_eval(y_true)
    y_pred = ast.literal_eval(y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    low_indices = y_true <= 70
    high_indices = y_true >= 180
    mid_indices = (y_true > 70) & (y_true < 180)

    low_score = metric(y_true[low_indices], y_pred[low_indices]) if low_indices.any() else None
    high_score = metric(y_true[high_indices],
                        y_pred[high_indices]) if high_indices.any() else None
    mid_score = metric(y_true[mid_indices], y_pred[mid_indices]) if mid_indices.any() else None

    # Compute the average score, weighting equally (only include valid scores)
    scores = [score for score in [low_score, high_score, mid_score] if score is not None]
    return scores


def get_scores_for_metric(df, ph, metric):
    y_true = df[f'target_{ph}'][0]
    y_pred = df[f'y_pred_{ph}'][0].replace("nan", "None")
    y_true = ast.literal_eval(y_true)
    y_pred = ast.literal_eval(y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if ph > 5:
        prev_y_true = df[f'target_{ph - 5}'][0]
        prev_y_true = ast.literal_eval(prev_y_true)
        delta_bgs = [val - prev_val for (val, prev_val) in zip(y_true, prev_y_true)]
    else:
        prev_y_true = df[f'CGM'][0]
        prev_y_true = ast.literal_eval(prev_y_true)
        delta_bgs = [val - prev_val for (val, prev_val) in zip(y_true, prev_y_true)]

    severity_weights = [slope_cost(bg, delta_bg) + zone_cost(bg) + 1 for bg, delta_bg in zip(y_true, delta_bgs)]

    # TODO: This solution gets a bit wrong for RMSE... I think it becomes MAE
    scores = [metric([true], [pred]) * severity_weight for true, pred, severity_weight in zip(y_true, y_pred, severity_weights)]
    return scores
