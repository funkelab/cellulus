import attrs

from .dataset_config import DatasetConfig
from .model_config import ModelConfig


@attrs.define
class InferenceConfig:
    """Inference configuration.

    model_config:

        The model configuration.

    inference_data_config:

        Configuration object for the data to predict and segment.
    """

    model_config: ModelConfig = attrs.field(converter=lambda d: ModelConfig(**d))
    inference_data_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )
