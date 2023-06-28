import click
import tomli

from cellulus.configs import ExperimentConfig
from cellulus.infer import infer as infer_experiment
from cellulus.train import train as train_experiment


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file):
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    train_experiment(ExperimentConfig(**config))


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def infer(config_file):
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    infer_experiment(ExperimentConfig(**config))
