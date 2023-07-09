from typing import Tuple

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig


@attrs.define
class InferenceConfig:
    """Inference configuration.

    dataset_config:

        Configuration object for the data to predict and segment.

    output_dataset_config:

        Configuration object produced by predict.py.

    crop_size:

        ROI used by the scan node in gunpowder.

    p_salt_pepper:

        Fraction of pixels that will have salt-pepper noise.

    num_infer_iterations:

        Number of times the salt-peper noise is added to the raw image.

    """

    dataset_config: DatasetConfig = attrs.field(converter=lambda d: DatasetConfig(**d))

    output_dataset_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )

    crop_size: Tuple = attrs.field(default=(252, 252), validator=instance_of(Tuple))
    p_salt_pepper = attrs.field(default=0.1, validator=instance_of(float))
    num_infer_iterations = attrs.field(default=16, validator=instance_of(int))
