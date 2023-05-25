import click
from constants import VALID_MODELS


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model-name", type=click.Choice(VALID_MODELS))
def train(model_name):
    pass


@cli.command()
@click.option("--model-file")
def evaluate(model_file):
    with open(model_file) as f:
        pass


if __name__ == "__main__":
    cli()

