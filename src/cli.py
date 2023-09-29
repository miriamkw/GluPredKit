


# data_storage.py or something like cli_helpers.py etc.
import csv
import pandas as pd

def store_data_as_csv(df, output_path, file_name):
    file_path = output_path + file_name
    df.to_csv(file_path)  # Set 'index' to False to exclude the index column


# cli.py
import click
import importlib
from parsers.base_parser import BaseParser
from datetime import timedelta, datetime


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout']), help='Choose a parser') # TODO: Make the list of parsers dynamic to the files in the parsers folder
@click.argument('username', type=str)
@click.argument('password', type=str)
@click.option('--file-name', type=click.Path(exists=True), help='Optional file name of output')
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
    start_date = end_date - timedelta(days=1)

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


if __name__ == "__main__":
    parse()
