from typing import List

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig


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

    crop_size:

        ROI used by the scan node in gunpowder.

    p_salt_pepper:

        Fraction of pixels that will have salt-pepper noise.

    num_infer_iterations:

        Number of times the salt-peper noise is added to the raw image.

    bandwidth:

        Band-width used to perform mean-shift clustering on the predicted
        embeddings.

    min_size:

        Ignore objects which are smaller than min_size number of pixels.

    """

    dataset_config: DatasetConfig = attrs.field(converter=lambda d: DatasetConfig(**d))

    prediction_dataset_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )

    segmentation_dataset_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )

    post_processed_dataset_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )
    crop_size: List = attrs.field(default=[252, 252], validator=instance_of(List))
    p_salt_pepper = attrs.field(default=0.1, validator=instance_of(float))
    num_infer_iterations = attrs.field(default=16, validator=instance_of(int))
    bandwidth = attrs.field(default=7, validator=instance_of(int))
    min_size = attrs.field(default=10, validator=instance_of(int))
    growd = attrs.field(default=3, validator=instance_of(int))
    threshold = attrs.field(default=6, validator=instance_of(int))
