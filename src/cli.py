import click
from constants import VALID_MODELS


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model-name", type=click.Choice(VALID_MODELS))
@click.option("--filename", default=None)
def train(model_name, filename):
    pass


@cli.command()
@click.option("--model-file")
def evaluate(model_file):
    with open(model_file) as f:
        pass

@cli.command()
@click.option("")

if __name__ == "__main__":
    cli()

