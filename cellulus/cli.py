import click
import tomli

from cellulus.configs import ExperimentConfig
from cellulus.predict import predict as predict_experiment
from cellulus.train import train as train_experiment


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file):
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    experiment_config = ExperimentConfig(**config)
    train_experiment(experiment_config)


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def predict(config_file):
    print(f"Reading config from {config_file}")
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    predict_config = ExperimentConfig(**config)
    predict_experiment(predict_config)
