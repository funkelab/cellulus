import os

import pytest
import tomli
from cellulus.configs import ExperimentConfig

this_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="module")
def experiment_config():
    """Create a dummy experiment config for the tests."""

    with open(os.path.join(this_dir, "train.toml"), "rb") as f:
        config = tomli.load(f)

    return ExperimentConfig(**config)
