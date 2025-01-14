import matplotlib.pyplot as plt
import ast
import numpy as np
import pandas as pd
import glupredkit.helpers.cli as helpers
from .base_plot import BasePlot


class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, dfs, prediction_horizon=120, zone_weighting_mode=2, *args):
        """
        Plots the confusion matrix for the given trained_models data.
        mode 0: normal metrics
        mode 1: using equally weighted scores
        mode 2: using cost function (severity blood glucose and gradient)
        """
        # TODO: Implement overall vs prediction horizon

        metrics = ['rmse', 'parkes_linear']
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

        num_models = len(dfs)
        num_metrics = len(metrics)

        x = np.arange(num_metrics)  # x-axis positions for metrics
        bar_width = 0.8 / num_models  # Width of each bar, leaving space between groups

        plt.figure(figsize=(12, 6))

        normalized_results = results_df.copy()
        metrics = normalized_results.columns[1:]  # Assuming first column is "Model Name"
        models = normalized_results["Model Name"].unique()

        # Normalize each metric
        for metric in metrics:
            min_val = 0.0
            max_val = np.abs(normalized_results[metric]).max()
            normalized_results[metric] = (np.abs(normalized_results[metric]) - min_val) / (max_val - min_val)

        for i, model in enumerate(models):
            y_values = normalized_results[normalized_results["Model Name"] == model][metrics].values.flatten()
            # y_values = [np.abs(val) for val in y_values]
            plt.bar(
                x + i * bar_width,  # Offset each model's bars
                y_values,
                width=bar_width,
                label=model,
                edgecolor="k",
                alpha=0.7,
            )

        # Customize the plot
        plt.xticks(x + bar_width * (num_models - 1) / 2, metrics, rotation=45, ha="right")  # Center tick labels
        plt.ylabel("Normalized Metric Score")
        plt.xlabel("Metric")
        plt.legend(title="Models")
        if len(prediction_horizons) < 1:
            plt.title(f"Metric Comparison Across Models for PH {prediction_horizon}")
        else:
            plt.title(f"Metric Comparison Across Models (Normalized) average across all prediction horizons")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Show the plot
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


