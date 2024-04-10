#!/usr/bin/env python3
import click
import dill
import os
import importlib
import ast
import pandas as pd
from datetime import timedelta, datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import numpy as np
import shap

# Modules from this repository
from .parsers.base_parser import BaseParser
from .plots.base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers.model_config_manager import ModelConfigurationManager, generate_model_configuration
import glupredkit.helpers.cli as helpers
import glupredkit.helpers.generate_report as generate_report


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

            merged_df = pd.DataFrame()

            for subject_id in ids_2018:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2018')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)

            for subject_id in ids_2020:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2020')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)
            save_data(output_file_name="OhioT1DM", data=merged_df)

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
                                            'tcn_pytorch',
                                            'loop',
                                            'zero_order',
                                            'naive_linear_regressor',
                                            'uva_padova'
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

    prediction_horizon = model_config_manager.get_prediction_horizon()
    model_module = helpers.get_model_module(model)

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
@click.argument('model_file', type=str)
def test_model(model_file):
    tested_models_path = "data/tested_models"

    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)
    training_data, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)
    target_cols = [col for col in test_data if col.startswith('target')]
    x_test = processed_data.drop(target_cols, axis=1)
    y_test = processed_data[target_cols]
    y_pred = model_instance.predict(x_test)

    hypo_threshold = 70
    hyper_threshold = 180

    # Create a dataframe to store the model name, configuration, predictions, and other results
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        'training_samples': [training_data.shape[0]],
        'test_samples': [test_data.shape[0]],
        'hypo_training_samples': [training_data[training_data['CGM'] < hypo_threshold].shape[0]],
        'hypo_test_samples': [test_data[test_data['CGM'] < hypo_threshold].shape[0]],
        'hyper_training_samples': [training_data[training_data['CGM'] > hyper_threshold].shape[0]],
        'hyper_test_samples': [test_data[test_data['CGM'] > hyper_threshold].shape[0]]
    })

    configs = model_config_manager.load_config()
    for config in configs:
        results_df[config] = [configs[config]]

    metrics = ['rmse', 'mare', 'me', 'parkes_error_grid', 'glycemia_detection', 'mcc_hypo', 'mcc_hyper']
    for i, minutes in enumerate(range(5, len(target_cols) * 5 + 1, 5)):
        curr_y_test = y_test[target_cols[i]].tolist()
        curr_y_pred = [val[i] for val in y_pred]
        results_df = results_df.copy()  # To silent PerformanceWarning
        results_df[target_cols[i]] = [curr_y_test]
        results_df[f'y_pred_{minutes}'] = [curr_y_pred]

        for metric in metrics:
            metric_module = helpers.get_metric_module(metric)
            chosen_metric = metric_module.Metric()
            score = chosen_metric(y_test[target_cols[i]], curr_y_pred)
            results_df[f'{metric}_{minutes}'] = [score]

    metrics = ['rmse', 'mare', 'me']
    # Calculate average error metrics
    for metric in metrics:
        metric_cols = [col for col in results_df.columns if col.startswith(metric)]
        avg_score = np.mean(results_df[metric_cols].iloc[0])
        results_df[f'{metric}_avg'] = [avg_score]

    insulin_dose = 1
    carb_intake = 50
    bolus_columns = [col for col in x_test.columns if col.startswith('bolus')]
    carbs_columns = [col for col in x_test.columns if col.startswith('carbs')]

    # TODO: Add a conditional check for the model, that decides how big the subset should be. 100 should be enough
    subset_size = x_test.shape[0]
    if model_name == 'loop':
        subset_size = 100
    subset_df_x = x_test.sample(n=subset_size, random_state=42)

    partial_dependency_dict = {}
    for col in bolus_columns:
        x_test_copy = subset_df_x.copy()
        x_test_copy[col] = insulin_dose
        y_pred = model_instance.predict(x_test_copy)
        # Calculate the average of elements at each index position
        averages = [sum(x) / len(x) for x in zip(*y_pred)]
        averages = [x - averages[0] for x in averages]
        partial_dependency_dict[col] = averages

    for col in carbs_columns:
        x_test_copy = subset_df_x.copy()
        x_test_copy[col] = carb_intake
        y_pred = model_instance.predict(x_test_copy)

        # Calculate the average of elements at each index position
        averages = [sum(x) / len(x) for x in zip(*y_pred)]
        averages = [x - averages[0] for x in averages]
        partial_dependency_dict[col] = averages

    results_df['partial_dependencies'] = [partial_dependency_dict]

    # Define the path to store the dataframe
    output_file = f"{tested_models_path}/{model_name}__{config_file_name}__{prediction_horizon}_results.csv"

    # Store the dataframe in a file
    results_df.to_csv(output_file, index=False)
    click.echo(f"Model {model_name} is finished testing. Results are stored in {tested_models_path}")


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
@click.option('--results-file', help='The name of the tested model results to evaluate, with ".csv".')
def generate_evaluation_pdf(results_file):
    """
    This command stores a standardized pdf report of the given model in data/reports/.
    """
    click.echo(f"Generating evaluation report...")
    df = generate_report.get_df_from_results_file(results_file)

    # Convert results to DataFrame and save as CSV
    results_file_path = 'data/reports/'
    os.makedirs(results_file_path, exist_ok=True)
    results_file_name = f'{results_file.split(".")[0]}.pdf'

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Evaluation for {df["Model Name"][0]}')
    c = generate_report.generate_single_model_front_page(c, df)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_accuracy_table(c, df)
    plot_height = 1.3  # TODO: Dynamically scale when table is larger
    c = generate_report.plot_across_prediction_horizons(c, df, f'RMSE [{unit_config_manager.get_unit()}]',
                                                        ['rmse'], y_placement=420, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'ME [{unit_config_manager.get_unit()}]',
                                                        ['me'], y_placement=260, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'MARE [%]',
                                                        ['mare'], y_placement=100, height=plot_height)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    c = generate_report.set_title(c, f'Error Grid Analysis')
    c = generate_report.draw_error_grid_table(c, df)
    c = generate_report.draw_scatter_plot(c, df, 30, 100, 360)
    c = generate_report.draw_scatter_plot(c, df, 60, 380, 360)
    prediction_horizon = generate_report.get_ph(df)
    c = generate_report.draw_scatter_plot(c, df, prediction_horizon - 30, 100, 120)
    c = generate_report.draw_scatter_plot(c, df, prediction_horizon, 380, 120)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION
    c = generate_report.set_title(c, f'Glycemia Detection - Matthews Correlation Coefficient (MCC)')
    c = generate_report.draw_mcc_table(c, df)
    c = generate_report.plot_across_prediction_horizons(c, df, f'Matthews Correlation Coefficient (MCC)',
                                                        ['mcc_hypo', 'mcc_hyper'], y_placement=150,
                                                        height=4, y_label='MCC',
                                                        y_labels=['MCC Hypoglycemia', 'MCC Hyperglycemia'])
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION PAGE 2
    c = generate_report.set_title(c, f'Glycemia Detection - Confusion Matrices')

    # Plot confusion matrixes
    classes = ['Hypo', 'Target', 'Hyper']

    c = generate_report.plot_confusion_matrix(c, df, classes, 30, 50, 450)
    c = generate_report.plot_confusion_matrix(c, df, classes, 60, 320, 450)
    c = generate_report.plot_confusion_matrix(c, df, classes, prediction_horizon - 50, 50, 150)
    c = generate_report.plot_confusion_matrix(c, df, classes, prediction_horizon, 320, 150)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PHYSIOLOGICAL ALIGNMENT PAGE
    c = generate_report.set_title(c, f'Physiological Alignment')
    c = generate_report.set_subtitle(c, f'Physiological Alignment for insulin', 720)
    c = generate_report.draw_physiological_alignment_table(c, df, 'bolus', 670)
    c = generate_report.plot_partial_dependency_heatmap(c, df, 'bolus', 100, 440,
                                                        f'Average impact on prediction for 1U of insulin')
    c = generate_report.set_subtitle(c, f'Physiological Alignment for carbohydrates', 380)
    c = generate_report.draw_physiological_alignment_table(c, df, 'carbs', 330)
    c = generate_report.plot_partial_dependency_heatmap(c, df, 'carbs', 100, 100,
                                                        f'Average impact on prediction for 50g of carbohydrates')
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PREDICTION DISTRIBUTION PAGE
    c = generate_report.set_title(c, f'Distribution of Predictions')
    c = generate_report.plot_predicted_distribution(c, df, 100, 400)

    # Show the page
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_file} is stored in '{results_file_path}' as '{results_file_name}'")


@click.command()
@click.option('--results-files', help='The name of the tested model results to evaluate, with ".csv". If '
                                      'None, all models will be tested.')
def generate_comparison_pdf(results_files):
    """
    This command stores a standardized pdf comparison report of the given models in data/reports/.
    """
    click.echo(f"Generating comparison report...")

    if results_files is None:
        results_files = helpers.list_files_in_directory('data/tested_models/')
    else:
        results_files = helpers.split_string(results_files)

    results_file_path = "data/reports/"
    results_file_name = f'comparison__{results_files}.pdf'

    dfs = []
    for results_file in results_files:
        df = generate_report.get_df_from_results_file(results_file)
        dfs += [df]

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Comparison Report')
    # TODO: Add front page summary of all parts
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_comparison_accuracy_table(c, dfs)
    c = generate_report.plot_rmse_across_prediction_horizons(c, dfs)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_files} is stored in '{results_file_path}' as '{results_file_name}'")


    """
    fig = plt.figure(figsize=(5, 3))

    for i in range(len(models)):
        # Plotting rmse values
        x_values = list(range(5, 5 * len(rmse_lists[i]) + 1, 5))
        plt.plot(x_values, rmse_lists[i], marker='o', label=f'{models[i]}')
        mean_rmse = np.mean(rmse_lists[i])
        plt.axhline(y=mean_rmse, color='black', linestyle='-')
        # Add text on the horizontal line
        plt.text(2, mean_rmse + 1, f'{models[i]}', color='black', fontsize=8)

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'RMSE [{unit_config_manager.get_unit()}]')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'RMSE [{unit_config_manager.get_unit()}]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 300)

    # Show the page
    c.showPage()

    # GLYCEMIA DETECTION
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Glycemia Detection")

    # Table data
    data = [
        ['Rank', 'Model', f'Hypoglycemia Detection', f'Hyperglycemia Detection', 'Overall']
    ]
    # Sort for ranking
    pairs = zip(glycemia_detection_list, [np.mean(values) for values in hypoglycemia_detection_lists],
                [np.mean(values) for values in hyperglycemia_detection_lists], models)

    # Sort the pairs based on the glycemia values (in descending order)
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Unpack the sorted pairs back into separate lists
    sorted_glycemia_list, sorted_hypo_list, sorted_hyper_list, sorted_models_list = zip(*sorted_pairs)

    for i in range(len(models)):
        glycemia_str = "{:.1f}%".format(sorted_glycemia_list[i])
        hypo_str = "{:.1f}%".format(sorted_hypo_list[i])
        hyper_str = "{:.1f}%".format(sorted_hyper_list[i])
        new_row = [f'#{i + 1}', sorted_models_list[i], hypo_str, hyper_str, glycemia_str]
        data += [new_row]

    # Create a table
    table = Table(data)
    table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('GRID', (0, 1), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0, colors.black)
    ]))

    # Draw the table on the canvas
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 100, 600)

    fig = plt.figure(figsize=(5, 2))

    for i in range(len(models)):
        # Plotting glycemia values
        x_values = list(range(5, 5 * len(rmse_lists[i]) + 1, 5))
        plt.plot(x_values, hypoglycemia_detection_lists[i], marker='o', label=f'{models[i]}')
        mean_hypo = np.mean(hypoglycemia_detection_lists[i])
        plt.axhline(y=mean_hypo, color='black', linestyle='-')
        # Add text on the horizontal line
        plt.text(2, mean_hypo + 1, f'{models[i]}', color='black', fontsize=8)

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Hypoglycemia Detection')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Hypoglycemia Detection [%]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 320)

    fig = plt.figure(figsize=(5, 2))

    for i in range(len(models)):
        # Plotting glycemia values
        x_values = list(range(5, 5 * len(rmse_lists[i]) + 1, 5))
        plt.plot(x_values, hyperglycemia_detection_lists[i], marker='o', label=f'{models[i]}')
        mean_hyper = np.mean(hyperglycemia_detection_lists[i])
        plt.axhline(y=mean_hyper, color='black', linestyle='-')
        # Add text on the horizontal line
        plt.text(2, mean_hyper + 1, f'{models[i]}', color='black', fontsize=8)

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Hyperglycemia Detection')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Hyperglycemia Detection [%]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 100)

    # Show the page
    c.showPage()

    # PREDICTION DISTRIBUTION
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Prediction Distribution")

    fig = plt.figure(figsize=(5, 3))

    for i in range(len(models)):
        x_values = list(range(5, 5 * len(rmse_lists[i]) + 1, 5))
        plt.plot(x_values, std_lists[i], marker='o', label=f'{models[i]}')

    plt.title(f'Prediction Standard Deviation')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Standard Deviation [{unit_config_manager.get_unit()}]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 450)

    fig = plt.figure(figsize=(5, 3))

    for i in range(len(models)):
        x_values = list(range(5, 5 * len(rmse_lists[i]) + 1, 5))
        plt.plot(x_values, std_diff_lists[i], marker='o', label=f'{models[i]}')

    plt.title(f'Prediction Standard Deviation minus Measurement Standard Deviation')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.ylabel(f'Standard Deviation [{unit_config_manager.get_unit()}]')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 100)

    # Show the page
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(
        f"An evaluation report for {models} is stored in '{results_file_path}' as '{results_file_name}'")
    """


@click.command()
@click.option('--use-mgdl', type=bool, help='Set whether to use mg/dL or mmol/L', default=None)
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
    'test_model': test_model,
    'calculate_metrics': calculate_metrics,
    'draw_plots': draw_plots,
    'generate_evaluation_pdf': generate_evaluation_pdf,
    'generate_comparison_pdf': generate_comparison_pdf,
    'set_unit': set_unit,
})

if __name__ == "__main__":
    cli()
