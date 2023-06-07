import click
import tomli
from cellulus.configs import ExperimentConfig


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file):
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    experiment_config = ExperimentConfig(**config)

    # TODO: start training


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def predict():
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    experiment_config = ExperimentConfig(**config)

    # TODO: start prediction
