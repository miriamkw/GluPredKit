#!/usr/bin/env python3
import click
import dill
import os
from io import BytesIO
import importlib
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

    prediction_range = int(prediction_horizon) // 5
    rmse_list = []
    rmse_module = helpers.get_metric_module("rmse")
    rmse = rmse_module.Metric()

    mae_list = []
    mae_module = helpers.get_metric_module("mae")
    mae = mae_module.Metric()

    clarke_error_grid_list = []
    clarke_module = helpers.get_metric_module("clarke_error_grid")
    clarke = clarke_module.Metric()

    hyperglycemia_detection_list = []
    hyper_module = helpers.get_metric_module("hyperglycemia_detection")
    hyper = hyper_module.Metric()

    hypoglycemia_detection_list = []
    hypo_module = helpers.get_metric_module("hypoglycemia_detection")
    hypo = hypo_module.Metric()

    for i in range(prediction_range):
        rmse_list += [rmse(y_test[target_columns[i]], y_pred[:, i])]
        mae_list += [mae(y_test[target_columns[i]], y_pred[:, i])]
        clarke_error_grid_list += [clarke(y_test[target_columns[i]], y_pred[:, i])]
        hyperglycemia_detection_list += [hyper(y_test[target_columns[i]], y_pred[:, i])]
        hypoglycemia_detection_list += [hypo(y_test[target_columns[i]], y_pred[:, i])]

    # Convert results to DataFrame and save as CSV
    results_file_path = 'data/reports/'
    os.makedirs(results_file_path, exist_ok=True)
    results_file_name = f'{model}.pdf'

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # Set font and font size for title
    c.setFont("Helvetica-Bold", 16)

    # Centered title
    c.drawCentredString(letter[0] / 2, 750, f'Model Evaluation for {model_name}')

    # Subtitle
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 720, f'Model Configuration')

    # Normal text
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f'Prediction horizon: {prediction_horizon} minutes')
    c.drawString(100, 680, f'Numerical features: [...]')
    c.drawString(100, 660, f'Categorical features: [...]')
    c.drawString(100, 640, f'What-if features: [...]')

    # Subtitle
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 600, f'Data Description')

    # Normal text
    c.setFont("Helvetica", 12)
    c.drawString(100, 580, f'Dataset: Name + citation')
    c.drawString(100, 560, f'Training samples: xxx')
    c.drawString(100, 540, f'Test samples: yyy')
    c.drawString(100, 520, f'Data variability index: zzz')
    c.drawString(100, 500, f'Hypoglcemia samples: zzz')
    c.drawString(100, 480, f'Hyperglycemia samples: zzz')

    # Subtitle
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 440, f'Model Evaluation Summary')

    # Normal text
    c.setFont("Helvetica", 12)
    c.drawString(100, 420, f'Summary of each part of the evaluation report...')

    # Bottom text
    c.setFont("Helvetica", 10)
    c.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                            f'(https://github.com/miriamkw/GluPredKit)')

    # Show the first page
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Evaluation Metrics")

    # Table data
    data = [
        ['Prediction Horizon', f'RMSE [{unit_config_manager.get_unit()}]', f'MAE [{unit_config_manager.get_unit()}]']
    ]

    for i in range(5, prediction_range, 6):
        rmse_str = "{:.1f}".format(rmse_list[i])
        mae_str = "{:.1f}".format(mae_list[i])
        new_row = [[str(i*5 + 5), rmse_str, mae_str]]
        data += new_row

    # Create a table
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    # Draw the table on the canvas
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 100, 600)

    # Plotting rmse values
    x_values = list(range(5, 5 * len(rmse_list) + 1, 5))
    fig = plt.figure(figsize=(5, 3))
    plt.plot(x_values, rmse_list, marker='o', label=f'RMSE [{unit_config_manager.get_unit()}]')
    plt.plot(x_values, mae_list, marker='o', label=f'MAE [{unit_config_manager.get_unit()}]')

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Error Metrics for {model_name}')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 300)

    # Subtitle
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 250, f'Clarke Error Grid Analysis')

    # Table data
    data = [
        ['Prediction Horizon', f'Zone A [%]', f'Zone B [%]', f'Zone C [%]', f'Zone D [%]', f'Zone E [%]']
    ]

    for i in range(5, prediction_range, 6):
        current_row = clarke_error_grid_list[i]
        new_row = [[str(i * 5 + 5), current_row[0], current_row[1], current_row[2], current_row[3], current_row[4]]]
        data += new_row

    # Create a table
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    # Draw the table on the canvas
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 100, 100)

    # Bottom text
    c.setFont("Helvetica", 10)
    c.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                            f'(https://github.com/miriamkw/GluPredKit)')

    # Show the second page
    c.showPage()

    # TODO: Reduce redundancy in this code
    # Page 3
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Scatter Plots")

    """
    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[5]], y_pred[:, 5], alpha=0.5)

    if unit_config_manager.use_mgdl:
        unit = "mg/dL"
        max_val = 400
    else:
        unit = "mmol/L"
        max_val = unit_config_manager.convert_value(400)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"30 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 100, 520)

    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[11]], y_pred[:, 11], alpha=0.5)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"60 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 400, 520)

    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[17]], y_pred[:, 17], alpha=0.5)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"90 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 100, 300)

    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[23]], y_pred[:, 23], alpha=0.5)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"120 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 400, 300)

    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[29]], y_pred[:, 29], alpha=0.5)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"150 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 100, 80)

    # Plotting rmse values
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(y_test[target_columns[35]], y_pred[:, 35], alpha=0.5)

    # Plotting the line x=y
    plt.plot([0, max_val], [0, max_val], 'k-')

    plt.xlabel(f"True Blood Glucose [{unit}]")
    plt.ylabel(f"Predicted Blood Glucose [{unit}]")
    plt.title(f"180 minutes")
    plt.legend(loc='upper left')

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 400, 80)
    
    """

    # Bottom text
    c.setFont("Helvetica", 10)
    c.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                            f'(https://github.com/miriamkw/GluPredKit)')

    # Show the page
    c.showPage()

    # Page 4
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Critical Clinical Aspects")

    # Normal text
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f'The detection of hypo- and hyperglycemia assesses the percentage of target ')
    c.drawString(100, 715, f'values below 70 mg/dL or above 180 mg/dL, where the predicted values also fall')
    c.drawString(100, 700, f'below 70 mg/dL or above 180 mg/dL.')

    # Table data
    data = [
        ['Prediction Horizon', f'Hypoglycemia Detection', f'Hyperglycemia Detection']
    ]

    for i in range(5, prediction_range, 6):
        hypoglycemia_str = "{:.1f}%".format(hypoglycemia_detection_list[i])
        hyperglycemia_str = "{:.1f}%".format(hyperglycemia_detection_list[i])
        new_row = [[str(i * 5 + 5), hypoglycemia_str, hyperglycemia_str]]
        data += new_row

    # Create a table
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    # Draw the table on the canvas
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 100, 550)

    # TODO: Add a plot for hypo- and hyperglycemia detection!

    # Plotting rmse values
    x_values = list(range(5, 5 * len(rmse_list) + 1, 5))
    fig = plt.figure(figsize=(5, 3))
    plt.plot(x_values, hypoglycemia_detection_list, marker='o', label=f'Hypoglycemia Detection [%]')
    plt.plot(x_values, hyperglycemia_detection_list, marker='o', label=f'Hyperglycemia Detection [%]')

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Glycemia Detection for {model_name}')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 250)


    # Bottom text
    c.setFont("Helvetica", 10)
    c.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                            f'(https://github.com/miriamkw/GluPredKit)')

    # Show the page
    c.showPage()

    # NEW PAGE
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(letter[0] / 2, 750, "Feature Impact on Blood Glucose Prediction Output")

    # Normal text
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f'The Partial Dependence Plots below illustrate how adjustments to model input ')
    c.drawString(100, 715, f'variables influence the model outputs across the predicted trajectory. ')
    c.drawString(100, 700, f'Each line on the plot represents the average predicted trajectory given a ')
    c.drawString(100, 685, f'specific input. In the physical reality, increased insulin should always decrease')
    c.drawString(100, 670, f'blood glucose, and increased carbohydrates should always increase blood glucose. ')

    # Plotting Partial Dependence Plots
    x_values = list(range(5, 5 * len(rmse_list) + 1, 5))
    fig = plt.figure(figsize=(5, 3))

    insulin_units = [0, 5, 10]
    test_column = 'insulin'

    for insulin_dose in insulin_units:
        x_test = processed_data.drop(target_columns, axis=1)
        x_test[test_column] = insulin_dose
        y_pred = model_instance.predict(x_test)
        # Calculate the average of elements at each index position
        averages = [sum(x) / len(x) for x in zip(*y_pred)]
        plt.plot(x_values, averages, marker='o', label=f'{insulin_dose} U')

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Partial Dependence Plot for {test_column}')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 390)

    # Plotting Partial Dependence Plots
    x_values = list(range(5, 5 * len(rmse_list) + 1, 5))
    fig = plt.figure(figsize=(5, 3))

    insulin_units = [0, 50, 100]
    test_column = 'carbs'

    for insulin_dose in insulin_units:
        x_test = processed_data.drop(target_columns, axis=1)
        x_test[test_column] = insulin_dose
        y_pred = model_instance.predict(x_test)
        # Calculate the average of elements at each index position
        averages = [sum(x) / len(x) for x in zip(*y_pred)]
        plt.plot(x_values, averages, marker='o', label=f'{insulin_dose} U')

    # Setting the title and labels with placeholders for the metric unit
    plt.title(f'Partial Dependence Plot for {test_column}')
    plt.xlabel('Prediction Horizons (minutes)')
    plt.legend()

    # Save the plot as an image
    buffer = BytesIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)  # Move the file pointer to the beginning
    drawing = svg2rlg(buffer)
    renderPDF.draw(drawing, c, 70, 100)


    # Bottom text
    c.setFont("Helvetica", 10)
    c.drawCentredString(letter[0] / 2, 50, f'This report is generated using GluPredKit '
                                           f'(https://github.com/miriamkw/GluPredKit)')

    # Show the page
    c.showPage()

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
