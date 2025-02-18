import os
import ast
import requests
import wandb
import numpy as np
import pandas as pd
import glupredkit.helpers.cli as helpers
from glupredkit.helpers.unit_config_manager import unit_config_manager
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
from matplotlib.figure import Figure


# DATA PARSING AND RETRIEVAL
def get_synthetic_data():
    url = 'https://raw.githubusercontent.com/miriamkw/GluPredKit/main/example_data/synthetic_data.csv'
    response = requests.get(url).content
    return pd.read_csv(BytesIO(response), index_col="date", parse_dates=True, low_memory=False)


def get_parsed_data(file_name):
    data = helpers.read_data_from_csv("data/raw/", file_name)
    return data


def features_target_split(processed_data):
    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x = processed_data.drop(target_columns, axis=1)
    y = processed_data[target_columns]
    return x, y


def get_results_df(model_name, training_data, test_data, y_pred, prediction_horizon, num_lagged_features, num_features,
                   cat_features, what_if_features, hypo_threshold=70, hyper_threshold=180):

    # Create a dataframe to store the model name, configuration, predictions, and other results
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        'training_samples': [training_data.shape[0]],
        'test_samples': [test_data.shape[0]],
        'hypo_training_samples': [training_data[training_data['CGM'] < hypo_threshold].shape[0]],
        'hypo_test_samples': [test_data[test_data['CGM'] < hypo_threshold].shape[0]],
        'hyper_training_samples': [training_data[training_data['CGM'] > hyper_threshold].shape[0]],
        'hyper_test_samples': [test_data[test_data['CGM'] > hyper_threshold].shape[0]],
        'unit': [unit_config_manager.get_unit()],
        'prediction_horizon': [prediction_horizon],
        'num_lagged_features': [num_lagged_features],
        'num_features': [num_features],
        'cat_features': [cat_features],
        'what_if_features': [what_if_features]
    })

    # Add daily average insulin if relevant
    if 'bolus' in num_features and 'basal' in num_features:
        test_data['insulin'] = test_data['bolus'] + (test_data['basal'] / 12)
        results_df['daily_avg_insulin'] = np.mean(test_data.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))
    elif ['insulin'] in num_features:
        results_df['daily_avg_insulin'] = np.mean(test_data.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))

    x_test, y_test = features_target_split(test_data)

    # Add test data input for numerical features
    for feature in num_features:
        results_df['test_input_' + feature] = [x_test[feature].tolist()]

    # Add test data dates
    results_df['test_input_date'] = [x_test.index.tolist()]

    metrics = helpers.list_files_in_package('metrics')
    metrics = [os.path.splitext(file)[0] for file in metrics if file not in ('__init__.py', 'base_metric.py')]

    target_cols = list(y_test.columns)
    if target_cols == ['target']:
        # In this case, targets are stored into sequences for Neural Networks
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_test['target']]
        targets = np.array(targets)
        _, n_predictions = targets.shape

        for i, minutes in enumerate(range(5, n_predictions * 5 + 1, 5)):
            curr_y_test = targets[:, i].tolist()
            curr_y_pred = [float(val[i]) for val in y_pred]
            results_df = results_df.copy()  # To silent PerformanceWarning
            results_df[f'target_{minutes}'] = [curr_y_test]
            results_df[f'y_pred_{minutes}'] = [curr_y_pred]

            for metric in metrics:
                metric_module = helpers.get_metric_module(metric)
                chosen_metric = metric_module.Metric()
                score = chosen_metric(curr_y_test, curr_y_pred, prediction_horizon=minutes)
                results_df[f'{metric}_{minutes}'] = [score]

    else:
        for i, minutes in enumerate(range(5, len(target_cols) * 5 + 1, 5)):
            curr_y_test = y_test[target_cols[i]].tolist()
            curr_y_pred = [float(val[i]) for val in y_pred]
            results_df = results_df.copy()  # To silent PerformanceWarning
            results_df[target_cols[i]] = [curr_y_test]
            results_df[f'y_pred_{minutes}'] = [curr_y_pred]

            for metric in metrics:
                metric_module = helpers.get_metric_module(metric)
                chosen_metric = metric_module.Metric()
                score = chosen_metric(y_test[target_cols[i]], curr_y_pred, prediction_horizon=minutes)
                results_df[f'{metric}_{minutes}'] = [score]

    for col in results_df.columns:
        if results_df[col].apply(lambda x: isinstance(x, list)).any():
            results_df[col] = results_df[col].astype(str)  # or df[col].apply(str)

    return results_df


def save_figures(figures, names):
    plot_results_path = get_figure_path()
    os.makedirs(plot_results_path, exist_ok=True)
    for current_plot, plot_name in zip(figures, names):
        is_valid = check_plot_validity(current_plot, plot_name)
        if is_valid:
            file_name = plot_name + '.png'
            print("Saving plot: ", file_name)
            current_plot.savefig(Path(plot_results_path, file_name))


def log_figures_in_wandb(wandb_project, figures, names):
    load_dotenv(".env.local")
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=wandb_api_key)
    wandb.init(
        project=wandb_project,
        name="glupredkit plots",
        tags="validation",
        job_type="eval"
    )
    for current_plot, plot_name in zip(figures, names):
        is_valid = check_plot_validity(current_plot, plot_name)
        if is_valid:
            wandb.log({plot_name: wandb.Image(current_plot)})


def get_figure_path():
    return Path('data', 'figures')



def check_plot_validity(plot, name):
    if not isinstance(plot, Figure):  # Matplotlib plot objects often return lists
        print(f"Error: Expected a plot, but got: {plot}")
        return False

    if not isinstance(name, str):
        print(f"Error: plot name is not a string. Current value: {name}")
        return False
    return True


