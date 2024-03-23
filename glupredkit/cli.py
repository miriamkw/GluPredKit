#!/usr/bin/env python3
import click
import dill
import os
import importlib
import pandas as pd
from datetime import timedelta, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Modules from this repository
from .parsers.base_parser import BaseParser
from .plots.base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers.model_config_manager import ModelConfigurationManager, generate_model_configuration
import glupredkit.helpers.cli as helpers


# TODO: Fix so that all default values are defined upstream (=here in the CLI), and removed from downstream


@click.command()
def setup_directories():
    """Set up necessary directories for GluPredKit."""
    cwd = os.getcwd()
    print("Creating directories...")

    folder_path = 'data'
    folder_names = ['raw', 'configurations', 'trained_models', 'figures', 'reports']

    for folder_name in folder_names:
        path = os.path.join(cwd, folder_path, folder_name)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory {path}.")

    print("Directories created for usage of GluPredKit.")


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout', 'apple_health', 'ohio_t1dm']),
              help='Choose a parser', required=True)
@click.option('--username', type=str, required=False)
@click.option('--password', type=str, required=False)
@click.option('--start-date', type=str,
              help='Start date for data retrieval. Default is two weeks ago. Format "dd-mm-yyyy"')
@click.option('--file-path', type=str, required=False)
@click.option('--subject-id', type=str, required=False)
@click.option('--end-date', type=str, help='End date for data retrieval. Default is now. Format "dd-mm-yyyy"')
@click.option('--output-file-name', type=str, help='The file name for the output.')
def parse(parser, username, password, start_date, file_path, subject_id, end_date, output_file_name):
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

    def save_data(output_file_name, data):
        output_path = 'data/raw/'
        date_format = "%d-%m-%Y"

        if output_file_name:
            file_name = output_file_name + '.csv'
        else:
            file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(
                date_format) + '.csv')

        click.echo("Storing data as CSV...")
        helpers.store_data_as_csv(data, output_path, file_name)
        click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
        click.echo(f"Data has the shape: {data.shape}")

    # Ensure that the optional params match the parser
    if parser in ['tidepool', 'nightscout']:
        if username is None or password is None:
            raise ValueError(f"{parser} parser requires that you provide --username and --password") 
        else:
            parsed_data = chosen_parser(start_date, end_date, username=username, password=password)
    elif parser in ['apple_health']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            parsed_data = chosen_parser(start_date, end_date, file_path=file_path)
    elif parser in ['ohio_t1dm']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            ids_2018 = ['559', '563', '570', '575', '588', '591']
            ids_2020 = ['540', '544', '552', '567', '584', '596']

            for subject_id in ids_2018:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2018')
                save_data(output_file_name=subject_id, data=parsed_data)

            for subject_id in ids_2020:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2020')
                save_data(output_file_name=subject_id, data=parsed_data)

            return
    else:
        raise ValueError(f"unrecognized parser: '{parser}'")

    save_data(output_file_name=output_file_name, data=parsed_data)



@click.command()
@click.option('--file-name', prompt='Configuration file name', help='Name of the configuration file.',
              callback=helpers.validate_file_name)
@click.option('--data', prompt='Input data file name (from data/raw/)', help='Name of the data file from data/raw/.',
              callback=helpers.validate_file_name)
@click.option('--preprocessor', prompt='Preprocessor (available: basic, ohio_t1dm)', help='Name of the preprocessor.')
@click.option('--prediction-horizons', prompt='Prediction horizons in minutes (comma-separated without space)',
              help='Comma-separated list of prediction horizons.', callback=helpers.validate_prediction_horizons)
@click.option('--num-lagged-features', prompt='Number of lagged features', help='Number of lagged features.',
              callback=helpers.validate_num_lagged_features)
@click.option('--num-features', prompt='Numerical features (a subset of column names from the input data file)',
              help='Comma-separated list of numerical features.', callback=helpers.validate_feature_list)
@click.option('--cat-features', prompt='Categorical features (press enter if none)', default='',
              help='Comma-separated list of categorical features.', callback=helpers.validate_feature_list)
@click.option('--test-size', prompt='Test size (float between 0 and 1)', callback=helpers.validate_test_size,
              help='Test size.')
def generate_config(file_name, data, preprocessor, prediction_horizons, num_lagged_features, num_features, cat_features,
                    test_size):
    generate_model_configuration(file_name, data, preprocessor, prediction_horizons, int(num_lagged_features),
                                 num_features, cat_features, float(test_size))
    click.echo(f"Storing configuration file to data/configurations/{file_name}...")
    click.echo(f"Note that it might take a minute before the file appears in the folder.")


@click.command()
@click.argument('model', type=click.Choice(['arx',
                                            'lstm',
                                            'lstm_pytorch',
                                            'random_forest',
                                            'ridge',
                                            'svr_linear',
                                            'svr_rbf',
                                            'tcn',
                                            'tcn_pytorch'
                                            ]))
@click.argument('config-file-name', type=str)
def train_model(model, config_file_name):
    """
    This method does the following:
    1) Process data using the given configurations
    2) Train the given models for the given prediction horizons in the configuration
    """
    click.echo(f"Starting pipeline to train model {model} with configurations in {config_file_name}...")
    model_config_manager = ModelConfigurationManager(config_file_name)

    prediction_horizons = model_config_manager.get_prediction_horizons()
    model_module = helpers.get_model_module(model)

    for prediction_horizon in prediction_horizons:
        # PREPROCESSING

        # Perform data preprocessing using your preprocessor
        train_data, _ = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)
        click.echo(f"Training data finished preprocessing...")

        # MODEL TRAINING
        # Create an instance of the chosen model
        chosen_model = model_module.Model(prediction_horizon)

        processed_data = chosen_model.process_data(train_data, model_config_manager, real_time=False)
        target_columns = [column for column in processed_data.columns if column.startswith('target')]
        x_train = processed_data.drop(target_columns, axis=1)
        y_train = processed_data[target_columns]

        click.echo(f"Training model...")

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
@click.option('--models', help='List of trained models separated by comma, without ".pkl". If none, all '
                               'models will be evaluated. ',
              default=None)
@click.option('--metrics', help='List of metrics to be computed, separated by comma. '
                                'By default RMSE will be computed. ', default='rmse')
def calculate_metrics(models, metrics):
    """
    This command stores a report of the given metrics in data/reports/.
    """
    trained_models_path = "data/trained_models/"

    if models is None:
        models = helpers.list_files_in_directory(trained_models_path)
    else:
        models = helpers.split_string(models)

    # Prepare a list of metrics
    metrics = helpers.split_string(metrics)

    results = []
    click.echo(f"Calculating metrics...")
    for model_file in models:
        model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                            int(model_file.split('__')[2].split('.')[0]))

        model_config_manager = ModelConfigurationManager(config_file_name)
        model_instance = helpers.get_trained_model(model_file)
        _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

        processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)
        x_test = processed_data.drop('target', axis=1)
        y_test = processed_data['target']

        y_pred = model_instance.predict(x_test)

        for metric in metrics:
            metric_module = helpers.get_metric_module(metric)
            chosen_metric = metric_module.Metric()
            score = chosen_metric(y_test, y_pred)
            results.append({'Model name': model_name,
                            'Configuration': config_file_name,
                            'Prediction horizon': prediction_horizon,
                            'Metric': metric,
                            'Score': score})

    # Convert results to DataFrame and save as CSV
    df_results = pd.DataFrame(results)
    metric_results_path = 'data/reports/'
    os.makedirs(metric_results_path, exist_ok=True)

    timestamp = datetime.now().isoformat()
    safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
    results_file_name = f'{safe_timestamp}.csv'

    df_results.to_csv(metric_results_path + results_file_name, index=False)

    click.echo(
        f"{metrics} for trained_models {models} are stored in '{metric_results_path}' as '{results_file_name}'")

    click.echo(df_results.to_string(index=False))


@click.command()
@click.option('--models', help='List of trained models separated by comma, without ".pkl". If none, all '
                               'models will be evaluated. ', default=None)
@click.option('--plots', help='List of plots to be computed, separated by comma. '
                              'By default a scatter plot will be drawn. ', default='scatter_plot')
@click.option('--is-real-time', type=bool, help='Whether to include test data without matching true measurements.'
    , default=False)
@click.option('--start-date', type=str,
              help='Start date for the predictions. Default is the first sample in the test data. '
                   'Format "dd-mm-yyyy/hh:mm"', default=None)
@click.option('--end-date', type=str,
              help='End date, or prediction date for one prediction plots. Default is the last sample in the test data.'
                   'Format "dd-mm-yyyy/hh:mm"', default=None)
@click.option('--carbs', type=int,
              help='Artificial carbohydrate input for one prediction plots. Only available when is-real-time is true.',
              default=None)
@click.option('--insulin', type=float,
              help='Artificial insulin input for one prediction plots. Only available when is-real-time is true.',
              default=None)
def draw_plots(models, plots, is_real_time, start_date, end_date, carbs, insulin):
    """
    This command draws the given plots and store them in data/figures/.

    The "real_time" parameter indicates whether all test data (including when there is no true values) should be
    included. This parameter allows for real-time plots, but is not compatible with all the plots.
    """
    # Prepare a list of plots
    plots = helpers.split_string(plots)

    if models is None:
        models = helpers.list_files_in_directory('data/trained_models/')
    else:
        models = helpers.split_string(models)

    click.echo(f"Calculating predictions...")
    models_data = []
    for model_file in models:
        model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                            int(model_file.split('__')[2].split('.')[0]))

        model_config_manager = ModelConfigurationManager(config_file_name)
        model_instance = helpers.get_trained_model(model_file)
        _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager, carbs=carbs,
                                                     insulin=insulin, start_date=start_date, end_date=end_date)

        processed_data = model_instance.process_data(test_data, model_config_manager, real_time=is_real_time)
        x_test = processed_data.drop('target', axis=1)
        y_test = processed_data['target']

        y_pred = model_instance.predict(x_test)

        model_data = {
            'name': f'{model_name} {prediction_horizon}-minutes PH',
            'y_pred': y_pred,
            'y_true': y_test,
            'prediction_horizon': prediction_horizon,
            'config': config_file_name,
            'real_time': is_real_time,
            'carbs': carbs,
            'insulin': insulin,
        }
        models_data.append(model_data)

    plot_results_path = 'data/figures/'

    # Draw plots
    click.echo(f"Drawing plots...")
    os.makedirs(plot_results_path, exist_ok=True)
    for plot in plots:
        plot_module = importlib.import_module(f'glupredkit.plots.{plot}')
        if not issubclass(plot_module.Plot, BasePlot):
            raise click.ClickException(f"The selected plot '{plot}' must inherit from BasePlot.")

        chosen_plot = plot_module.Plot()
        chosen_plot(models_data)

    click.echo(f"{plots} for trained_models {models} are stored in '{plot_results_path}'")


@click.command()
@click.option('--model', help='The name of the trained model to evaluate, without ".pkl".')
def generate_evaluation_pdf(model):
    """
    This command stores a standardized pdf report of the given model in data/reports/.
    """
    trained_models_path = "data/trained_models/"

    click.echo(f"Generating evaluation report...")
    model_name, config_file_name, prediction_horizon = (model.split('__')[0], model.split('__')[1],
                                                        int(model.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(f'{model}.pkl')
    _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)

    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x_test = processed_data.drop(target_columns, axis=1)
    y_test = processed_data[target_columns]

    y_pred = model_instance.predict(x_test)

    # Convert results to DataFrame and save as CSV
    results_file_path = 'data/reports/'
    os.makedirs(results_file_path, exist_ok=True)
    results_file_name = f'{model}.pdf'

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)
    # Draw "Hello World" text
    c.drawString(100, 750, "Hello World")
    # Save the PDF
    c.save()

    click.echo(
        f"An evaluation report for {model}.pkl is stored in '{results_file_path}' as '{results_file_name}'")



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
    'generate_config': generate_config,
    'train_model': train_model,
    'calculate_metrics': calculate_metrics,
    'draw_plots': draw_plots,
    'generate_evaluation_pdf': generate_evaluation_pdf,
    'set_unit': set_unit,
})

if __name__ == "__main__":
    cli()
