from typing import List

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig
from .utils import to_config


@attrs.define
class SemiSupervisedTrainConfig:
    """Train configuration.

    Parameters:

        raw_data_config:

            Configuration object for the raw training data.

        pseudo_data_config:

            Configuration object for the pseudo-ground-truth labels.

        supervised_data_config:

            Configuration object for the ground-truth labels/annotations.

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

        reduce_mean (default = True):

            If True, the loss contribution is averaged across all pairs of patches.

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

        control_point_spacing (default = 64):

            The distance in pixels between control points used for elastic
            deformation of the raw data during training.

        control_point_jitter (default = 2.0):

            How much to jitter the control points for elastic deformation
            of the raw data during training, given as the standard deviation of
            a normal distribution with zero mean.

        
    """

    raw_data_config: DatasetConfig = attrs.field(converter=to_config(DatasetConfig))
    pseudo_data_config: DatasetConfig = attrs.field(converter=to_config(DatasetConfig))
    supervised_data_config: DatasetConfig = attrs.field(converter=to_config(DatasetConfig))

    crop_size: List = attrs.field(default=[252, 252], validator=instance_of(List))
    batch_size: int = attrs.field(default=8, validator=instance_of(int))
    max_iterations: int = attrs.field(default=100_000, validator=instance_of(int))
    initial_learning_rate: float = attrs.field(
        default=4e-5, validator=instance_of(float)
    )
    density: float = attrs.field(default=0.2, validator=instance_of(float))
    kappa: float = attrs.field(default=10.0, validator=instance_of(float))
    temperature: float = attrs.field(default=10.0, validator=instance_of(float))
    regularizer_weight: float = attrs.field(default=1e-5, validator=instance_of(float))
    reduce_mean: bool = attrs.field(default=True, validator=instance_of(bool))
    save_model_every: int = attrs.field(default=1_000, validator=instance_of(int))
    save_snapshot_every: int = attrs.field(default=1_000, validator=instance_of(int))
    num_workers: int = attrs.field(default=8, validator=instance_of(int))

    control_point_spacing: int = attrs.field(default=64, validator=instance_of(int))
    control_point_jitter: float = attrs.field(default=2.0, validator=instance_of(float))
