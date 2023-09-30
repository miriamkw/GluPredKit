import click
import importlib
import pandas as pd
from parsers.base_parser import BaseParser
from preprocessors.base_preprocessor import BasePreprocessor
from datetime import timedelta, datetime


def read_data_from_csv(input_path, file_name):
    file_path = input_path + file_name
    return pd.read_csv(file_path, index_col="date", parse_dates=True)


def store_data_as_csv(df, output_path, file_name):
    file_path = output_path + file_name
    df.to_csv(file_path)


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout']), help='Choose a parser') # TODO: Make the list of parsers dynamic to the files in the parsers folder
@click.argument('username', type=str)
@click.argument('password', type=str)
@click.option('--file-name', type=str, help='Optional file name of output')
# TODO: Add start and end dates as input
def parse(parser, username, password, file_name):
    """Parse data and store it as CSV in data/raw using a selected parser"""

    # Load the chosen parser dynamically based on user input
    parser_module = importlib.import_module(f'parsers.{parser}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(parser_module.Parser, BaseParser):
        raise click.ClickException(f"The selected parser '{parser}' must inherit from BaseParser.")

    # Create an instance of the chosen parser
    chosen_parser = parser_module.Parser()

    click.echo(f"Parsing data using {parser}...")

    end_date = datetime.now()
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=14)

    # Perform parsing using the chosen parser
    parsed_data = chosen_parser(start_date, end_date, username, password)

    output_path = '../data/raw/'
    date_format = "%d-%m-%Y"

    # Add default file name if input is not provided
    if file_name is not None:
        file_name = file_name
    else:
        file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(date_format)
                     + '.csv')

    click.echo("Storing data as CSV...")
    store_data_as_csv(parsed_data, output_path, file_name)
    click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
    click.echo(f"Data has the shape: {parsed_data.shape}")


# TODO: Preprocessor alternatives: prediction_horizon=60, num_lagged_features=12, include_hour=True, test_size=0.2, optional file name

@click.command()
@click.option('--preprocessor', type=click.Choice(['scikit_learn', 'other']), help='Choose a preprocessor') # TODO: Make the list of parsers dynamic to the files in the parsers folder
@click.argument('input_file_name', type=str)
# @click.option('--test-size', type=float, default=0.2, help='Fraction of data to reserve for testing (default: 0.2)')
def preprocess(preprocessor, input_file_name):
    """
    Preprocess data from an input CSV file and store train and test data into CSV files.

    Args:
        preprocessor (str): Type of preprocessor from the preprocessor module.
        input_file_name (str): Input CSV file containing the data.
        test_size (float): Fraction of data to reserve for testing (default: 0.2).
    """

    # Load the chosen parser dynamically based on user input
    preprocessor_module = importlib.import_module(f'preprocessors.{preprocessor}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(preprocessor_module.Preprocessor, BasePreprocessor):
        raise click.ClickException(f"The selected preprocessor '{preprocessor}' must inherit from BasePreprocessor.")

    # Create an instance of the chosen parser
    chosen_preprocessor = preprocessor_module.Preprocessor()

    input_path = "../data/raw/"
    click.echo(f"Preprocessing data using {preprocessor} from file {input_path}{input_file_name}...")

    # Load the input CSV file into a DataFrame
    data = read_data_from_csv(input_path, input_file_name)

    # Perform data preprocessing using your preprocessor
    train_data, test_data = chosen_preprocessor(data)

    # Define output file names
    # TODO: add more descriptive names
    output_path = "../data/processed/"
    train_output_file = f"train_data.csv"
    test_output_file = f"test_data.csv"

    # Store train and test data as CSV files
    store_data_as_csv(train_data, output_path, train_output_file)
    store_data_as_csv(test_data, output_path, test_output_file)

    click.echo(f"Train data saved as '{train_output_file}', with shape {train_data.shape}")
    click.echo(f"Test data saved as '{test_output_file}', with shape {test_data.shape}")


if __name__ == "__main__":
    # Create a Click group and add the commands to it
    cli = click.Group(commands={
        'parse': parse,
        'preprocess': preprocess
    })

    cli()
