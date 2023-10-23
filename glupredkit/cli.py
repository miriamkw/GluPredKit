#!/usr/bin/env python3
import click
import dill
import os
import json
import importlib
import pandas as pd
from datetime import timedelta, datetime

# Modules from this repository
from .parsers.base_parser import BaseParser
from .preprocessors.base_preprocessor import BasePreprocessor
from .models.base_model import BaseModel
from .metrics.base_metric import BaseMetric
from .plots.base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
import glupredkit.helpers.cli as helpers



# TODO: Fix so that all default values are defined upstream (=here in the CLI), and removed from downstream


@click.command()
def setup_directories():
    """Set up necessary directories for GluPredKit."""
    cwd = os.getcwd()
    print("Creating directories...")

    folder_path = 'data'
    folder_names = ['raw', 'processed', 'trained_models', 'figures', 'reports']

    for folder_name in folder_names:
        path = os.path.join(cwd, folder_path, folder_name)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory {path}.")

    print("Directories created for usage of GluPredKit.")


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout', 'apple_health']), help='Choose a parser',
              required=True)
@click.argument('username', type=str)
@click.argument('password', type=str)
@click.option('--start-date', type=str,
              help='Start date for data retrieval. Default is two weeks ago. Format "dd-mm-yyyy"')
@click.option('--end-date', type=str,
              help='End date for data retrieval. Default is now. Format "dd-mm-yyyy"')
def parse(parser, username, password, start_date, end_date):
    """Parse data and store it as CSV in data/raw using a selected parser"""

    # Load the chosen parser dynamically based on user input
    parser_module = importlib.import_module(f'glupredkit.parsers.{parser}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(parser_module.Parser, BaseParser):
        raise click.ClickException(f"The selected parser '{parser}' must inherit from BaseParser.")

    # Create an instance of the chosen parser
    chosen_parser = parser_module.Parser()

    click.echo(f"Parsing data using {parser}...")

    date_format = "%d-%m-%Y"

    if not end_date:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, date_format)

    if not start_date:
        start_date = end_date - timedelta(days=14)
    else:
        start_date = datetime.strptime(start_date, date_format)

    # Perform parsing using the chosen parser
    parsed_data = chosen_parser(start_date, end_date, username, password)

    output_path = 'data/raw/'
    date_format = "%d-%m-%Y"

    file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(date_format)
                 + '.csv')

    click.echo("Storing data as CSV...")
    helpers.store_data_as_csv(parsed_data, output_path, file_name)
    click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
    click.echo(f"Data has the shape: {parsed_data.shape}")


@click.command()
@click.option('--preprocessor', type=click.Choice(['scikit_learn', 'tf_keras', 'basic']),
              help='Choose a preprocessor', required=True)
@click.argument('input-file-name', type=str)
@click.option('--prediction-horizon', type=int, default=60)
@click.option('--num-lagged-features', type=int, default=12,
              help='The number of samples of time-lagged features (default: 12).')
@click.option('--test-size', type=float, default=0.2,
              help='Fraction of data to reserve for testing (default: 0.2).')
@click.option('--num-features', default='CGM,insulin,carbs', help='List of numerical features, separated by comma.')
@click.option('--cat-features', default='', help='List of categorical features, separated by comma.')
def preprocess(preprocessor, input_file_name, prediction_horizon, num_lagged_features, test_size,
               num_features, cat_features):
    """
    Preprocess data from an input CSV file and store train and test data into CSV files.

    Args:
        preprocessor (str): Type of preprocessor from the preprocessor module.
        input_file_name (str): Input CSV file containing the data.
        prediction_horizon (int): The prediction horizon for the target value in minutes.
        num_lagged_features (int): The number of samples of time-lagged features.
        test_size (float): Fraction of data to reserve for testing.
        num_features (str): List of numerical features, separated with comma without spaces
        cat_features (str): List of categorical features, separated with comma without spaces
    """
    if prediction_horizon % 5 != 0:
        raise click.BadParameter('Prediction horizon must be divisible by 5.')

    # Load the chosen parser dynamically based on user input
    preprocessor_module = importlib.import_module(f'glupredkit.preprocessors.{preprocessor}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(preprocessor_module.Preprocessor, BasePreprocessor):
        raise click.ClickException(f"The selected preprocessor '{preprocessor}' must inherit from BasePreprocessor.")

    # Convert comma-separated string of features to list
    num_features = helpers.split_string(num_features)
    cat_features = helpers.split_string(cat_features)

    # Create an instance of the chosen parser
    chosen_preprocessor = preprocessor_module.Preprocessor(num_features, cat_features, prediction_horizon,
                                                           num_lagged_features, test_size)

    input_path = "data/raw/"
    click.echo(f"Preprocessing data using {preprocessor} from file {input_path}{input_file_name}...")

    # Load the input CSV file into a DataFrame
    data = helpers.read_data_from_csv(input_path, input_file_name)

    # Perform data preprocessing using your preprocessor
    train_data, test_data = chosen_preprocessor(data)

    # Define output file names
    output_path = "data/processed/"
    train_output_file = f"train-data_{preprocessor}_ph-{prediction_horizon}_lag-{num_lagged_features}.csv"
    test_output_file = f"test-data_{preprocessor}_ph-{prediction_horizon}_lag-{num_lagged_features}.csv"

    # Store train and test data as CSV files
    helpers.store_data_as_csv(train_data, output_path, train_output_file)
    helpers.store_data_as_csv(test_data, output_path, test_output_file)

    click.echo(f"Train data saved as '{train_output_file}', with shape {train_data.shape}")
    click.echo(f"Test data saved as '{test_output_file}', with shape {test_data.shape}")


@click.command()
@click.option('--model', prompt='Model name', help='Name of the model file (without .py) to be trained.')
@click.argument('config-file-name', type=str)
def train_model_new(model, config_file_name):
    """
    This method does the following:
    1) Process data using the given configurations
    2) Train the given models for the given prediction horizons in the configuration
    """
    input_path = "data/configurations/"
    click.echo(f"Starting pipeline to train model {model} with configurations from {input_path}{config_file_name}...")

    with open(f"{input_path}{config_file_name}.json") as f:
        config = json.load(f)

    prediction_horizons = config['prediction_horizons']
    model_module = helpers.get_model_module(model)

    for prediction_horizon in prediction_horizons:
        # PREPROCESSING

        # Perform data preprocessing using your preprocessor
        train_data, _ = helpers.get_preprocessed_data(prediction_horizon, config)

        # MODEL TRAINING
        # Create an instance of the chosen model
        chosen_model = model_module.Model(prediction_horizon)

        processed_data = chosen_model.process_data(train_data, config['num_lagged_features'], config['num_features'],
                                                   config['cat_features'])

        x_train = processed_data.drop('target', axis=1)
        y_train = processed_data['target']

        # Initialize and train the model
        model_instance = chosen_model.fit(x_train, y_train)
        click.echo(f"Model {model} with prediction horizon {prediction_horizon} minutes trained successfully!")

        # Assuming model_instance is your class instance
        output_path = "data/trained_models/"
        output_file_name = f'{model}__{config_file_name}__{prediction_horizon}.pkl'
        try:
            with open(f'{output_path}{output_file_name}', 'wb') as f:
                click.echo(f"Saving model {model} to {output_path}{output_file_name}...")
                dill.dump(model_instance, f)
        except Exception as e:
            click.echo(f"Error saving model {model}: {e}")

        if hasattr(model_instance, 'best_params'):
            click.echo(f"Model hyperparameters: {model_instance.best_params()}")



@click.command()
@click.option('--model', prompt='Model name', help='Name of the model file (without .py) to be trained.')
@click.argument('input-file-name', type=str)
@click.option('--prediction-horizon', type=int, default=60)
def train_model(model, input_file_name, prediction_horizon):
    # Load the chosen parser dynamically based on user input
    model_module = importlib.import_module(f'glupredkit.models.{model}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(model_module.Model, BaseModel):
        raise click.ClickException(f"The selected model '{model}' must inherit from BaseModel.")

    if input_file_name.startswith("test"):
        helpers.user_input_prompt("This file starts with 'test', and not 'train'. Are you sure you want to continue?")

    if str(prediction_horizon) not in input_file_name:
        helpers.user_input_prompt(f"The prediction horizon '{str(prediction_horizon)}' is not in the input file name. "
                          "Are you sure you want to continue?")

    # Create an instance of the chosen model
    chosen_model = model_module.Model(prediction_horizon)

    input_path = "data/processed/"
    click.echo(f"Training model {model} with training data from {input_path}{input_file_name}...")

    # Load the input CSV file into a DataFrame
    train_data = helpers.read_data_from_csv(input_path, input_file_name)

    x_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    # Initialize and train the model
    model_instance = chosen_model.fit(x_train, y_train)
    click.echo(f"Model {model} trained successfully!")

    # Assuming model_instance is your class instance
    output_path = "data/trained_models/"
    output_file_name = f'{model}_ph-{prediction_horizon}.pkl'
    try:
        with open(f'{output_path}{output_file_name}', 'wb') as f:
            click.echo(f"Saving model {model} to {output_path}{output_file_name}...")
            dill.dump(model_instance, f)
    except Exception as e:
        click.echo(f"Error saving model {model}: {e}")

    if hasattr(model_instance, 'best_params'):
        click.echo(f"Model hyperparameters: {model_instance.best_params()}")


@click.command()
@click.option('--model-files', prompt='Model file names',
              help='List of trained trained_models (filenames without .pkl), separated by comma. ')
@click.option('--metrics', help='List of metrics to be computed, separated by comma. '
                                'By default all metrics will be computed. ', default='mae,rmse,pcc')
@click.option('--plots', help='List of plots to be computed, separated by comma. '
                              'By default a scatter plot will be drawn. ', default='scatter_plot')
@click.argument('test-file-name', type=str)
@click.option('--prediction-horizon', type=int, default=60)
def evaluate_model(model_files, metrics, plots, test_file_name, prediction_horizon):
    """
    This command is only capable of comparing metrics given one specific prediction horizon at a time, because
    it only takes in one test file at a time. Hence, giving correct information about prediction horizon and
    test file is crucial.
    """
    if test_file_name.startswith("train"):
        helpers.user_input_prompt("This file starts with 'train', and not 'test'. Are you sure you want to continue?")

    if str(prediction_horizon) not in test_file_name:
        helpers.user_input_prompt(f"The prediction horizon '{str(prediction_horizon)}' is not in the input file name. "
                          "Are you sure you want to continue?")

    model_path = "data/trained_models/"
    metric_results_path = "data/reports/"
    plot_results_path = "data/figures/"
    test_file_path = "data/processed/"

    # Prepare a list of trained_models
    trained_models = helpers.split_string(model_files)

    # Prepare a list of metrics
    metrics = helpers.split_string(metrics)

    # Prepare a list of plots
    plots = helpers.split_string(plots)

    test_data = helpers.read_data_from_csv(test_file_path, test_file_name)
    x_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    results = []
    y_preds = []
    click.echo(f"Calculating metrics...")
    for model_file in trained_models:
        if str(prediction_horizon) not in model_file:
            helpers.user_input_prompt(f"The prediction horizon '{str(prediction_horizon)}' is not in the model file name. "
                              "Are you sure you want to continue?")

        with open(model_path + model_file + '.pkl', 'rb') as f:
            model_instance = dill.load(f)

        y_pred = model_instance.predict(x_test)
        y_preds.append(y_pred)

        for metric in metrics:
            metric_module = importlib.import_module(f'glupredkit.metrics.{metric}')
            if not issubclass(metric_module.Metric, BaseMetric):
                raise click.ClickException(f"The selected metric '{metric}' must inherit from BaseMetric.")

            chosen_metric = metric_module.Metric()
            score = chosen_metric(y_test, y_pred)
            results.append({'Model': model_file, 'Metric': metric, 'Score': score})

    # Convert results to DataFrame and save as CSV
    df_results = pd.DataFrame(results)
    os.makedirs(metric_results_path, exist_ok=True)
    results_file_name = f'{test_file_name}'
    df_results.to_csv(metric_results_path + results_file_name, index=False)

    click.echo(
        f"{metrics} for trained_models {model_files} are stored in '{metric_results_path}' as '{results_file_name}'")

    models_data = []
    for model_name, y_pred in zip(trained_models, y_preds):
        model_data = {
            'name': model_name,
            'y_pred': y_pred
        }
        models_data.append(model_data)

    # Draw plots
    click.echo(f"Drawing plots...")
    os.makedirs(plot_results_path, exist_ok=True)
    for plot in plots:
        plot_module = importlib.import_module(f'glupredkit.plots.{plot}')
        if not issubclass(plot_module.Plot, BasePlot):
            raise click.ClickException(f"The selected plot '{plot}' must inherit from BasePlot.")

        chosen_plot = plot_module.Plot(prediction_horizon)
        chosen_plot(models_data, y_test)

    click.echo(f"{plots} for trained_models {trained_models} are stored in '{plot_results_path}'")


@click.command()
@click.option('--use-mgdl', type=bool, help='Set whether to use mg/dL or mmol/L')
def set_unit(use_mgdl):
    # Update the config
    unit_config_manager.use_mgdl = use_mgdl
    click.echo(f'Set unit to {"mg/dL" if use_mgdl else "mmol/L"}.')


# Create a Click group and add the commands to it
cli = click.Group(commands={
    'setup_directories': setup_directories,
    'parse': parse,
    'preprocess': preprocess,
    'train_model': train_model,
    'train_model_new': train_model_new,
    'evaluate_model': evaluate_model,
    'set_unit': set_unit,
})

if __name__ == "__main__":
    cli()
