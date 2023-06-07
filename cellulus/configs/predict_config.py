import attrs

from .dataset_config import DatasetConfig
from .model_config import ModelConfig


@attrs.define
class PredictConfig:
    """Predict configuration.

    model_config:

        The model configuration.

    predict_data_config:

        Configuration object for the data to predict on.
    """

    model_config: ModelConfig = attrs.field(converter=lambda d: ModelConfig(**d))
    predict_data_config: DatasetConfig = attrs.field(
        converter=lambda d: DatasetConfig(**d)
    )
