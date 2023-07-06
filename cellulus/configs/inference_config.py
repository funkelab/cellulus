from typing import Tuple

import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig


@attrs.define
class InferenceConfig:
    """Inference configuration.

    dataset_config:

        Configuration object for the data to predict and segment.
    """

    dataset_config: DatasetConfig = attrs.field(converter=lambda d: DatasetConfig(**d))

    output_dataset_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )

    crop_size: Tuple = attrs.field(default=(252, 252), validator=instance_of(Tuple))
