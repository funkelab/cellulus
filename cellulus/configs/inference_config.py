from typing import List

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig
from .utils import to_config


@attrs.define
class InferenceConfig:
    """Inference configuration.

    dataset_config:

        Configuration object for the data to predict and segment.

    prediction_dataset_config:

        Configuration object produced by predict.py.

    segmentation_dataset_config:

        Configuration object produced by segment.py.

    post_processed_dataset_config:

        Configuration object produced by postprocess.py.


    evaluation_dataset_config:

        Configuration object for the ground truth masks.

    crop_size:

        ROI used by the scan node in gunpowder.

    p_salt_pepper:

        Fraction of pixels that will have salt-pepper noise.

    num_infer_iterations:

        Number of times the salt-peper noise is added to the raw image.

    bandwidth:

        Band-width used to perform mean-shift clustering on the predicted
        embeddings.

    reduction_probability:



    min_size:

        Ignore objects which are smaller than min_size number of pixels.

    device (default = 'cuda:0'):

            The device to infer on.
            Set to 'cpu' to infer without GPU.

    num_bandwidths (default = 1):

        Number of bandwidths to obtain segmentations for.

    """

    dataset_config: DatasetConfig = attrs.field(converter=lambda d: DatasetConfig(**d))

    prediction_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    segmentation_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    post_processed_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    evaluation_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )
    device: str = attrs.field(default="cuda:0", validator=instance_of(str))
    crop_size: List = attrs.field(default=[252, 252], validator=instance_of(List))
    p_salt_pepper = attrs.field(default=0.01, validator=instance_of(float))
    num_infer_iterations = attrs.field(default=16, validator=instance_of(int))
    bandwidth = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(int))
    )
    num_bandwidths = attrs.field(default=1, validator=instance_of(int))
    reduction_probability = attrs.field(default=0.1, validator=instance_of(float))
    min_size = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(int))
    )
    grow_distance = attrs.field(default=3, validator=instance_of(int))
    shrink_distance = attrs.field(default=6, validator=instance_of(int))
