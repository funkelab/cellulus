import attrs

from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from attrs.validators import instance_of



@attrs.define
class InferenceConfig:
    """Inference configuration.

    model_config:

        The model configuration.

    state_dict_path:

        The path to the torch state_dict for the trained model used for inference.

    inference_data_config:

        Configuration object for the data to predict and segment.
    """

    model_config: ModelConfig = attrs.field(converter=lambda d: ModelConfig(**d))
    state_dict_path: str = attrs.field(validator=instance_of(str))
    inference_data_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )
