from typing import Tuple

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig
from .utils import to_config


@attrs.define
class TrainConfig:
    """Train configuration.

    Parameters:

        crop_size:

            The size of the crops - specified as a tuple of pixels -
            extracted from the raw images, used during training.

        batch_size:

            The number of samples to use per batch.

        max_iterations:

            The maximum number of iterations to train for.

        initial_learning_rate (default = 4e-5):

            Initial learning rate of the optimizer.

        temperature (default = 10):

            Factor used to scale the gaussian function and control the rate of damping.

        regularizer_weight (default = 1e-5):

            The weight of the L2 regularizer on the object-centric embeddings.

        density (default = 0.2)

            Determines the fraction of patches to sample per crop, during training.

        kappa (default = 10.0):

            Neighborhood radius to extract patches from

        save_model_every (default = 1e3):

            The model weights are saved every few iterations.

        save_snapshot_every (default = 1e3):

            The zarr snapshot is saved every few iterations.

        num_workers (default = 8):

            The number of sub-processes to use for data-loading.

        train_data_config:

            Configuration object for the training data.

        validate_data_config:

            Configuration object for the validation data.
    """

    train_data_config: DatasetConfig = attrs.field(converter=to_config(DatasetConfig))
    validate_data_config: DatasetConfig = attrs.field(
        converter=to_config(DatasetConfig)
    )
    crop_size: Tuple = attrs.field(default=(252, 252), validator=instance_of(Tuple))
    batch_size: int = attrs.field(default=8, validator=instance_of(int))
    max_iterations: int = attrs.field(default=100_000, validator=instance_of(int))
    initial_learning_rate: float = attrs.field(
        default=4e-5, validator=instance_of(float)
    )
    density: float = attrs.field(default=0.2, validator=instance_of(float))
    kappa: float = attrs.field(default=10.0, validator=instance_of(float))
    temperature: float = attrs.field(default=10.0, validator=instance_of(float))
    regularizer_weight: float = attrs.field(default=1e-5, validator=instance_of(float))
    save_model_every: int = attrs.field(default=1_000, validator=instance_of(int))
    save_snapshot_every: int = attrs.field(default=1_000, validator=instance_of(int))
    num_workers: int = attrs.field(default=8, validator=instance_of(int))
