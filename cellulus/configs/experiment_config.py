import attrs
from attrs.validators import instance_of

from .model_config import ModelConfig
from .inference_config import InferenceConfig
from .train_config import TrainConfig
from .utils import to_config


@attrs.define
class ExperimentConfig:
    """Top-level config for an experiment (containing training and prediction).

    Parameters:

        experiment_name:

            A unique name for the experiment.

        object_size:

            A rough estimate of the size of objects in the image, given in
            world units. The "patch size" of the network will be chosen based
            on this estimate.

        model_config:

            The model configuration.

        train_config:

            Configuration object for training.

        inference_config:

            Configuration object for prediction.
    """

    experiment_name: str = attrs.field(validator=instance_of(str))
    object_size: float = attrs.field(validator=instance_of(float))

    model_config: ModelConfig = attrs.field(converter=to_config(ModelConfig))
    train_config: TrainConfig = attrs.field(
        default=None, converter=to_config(TrainConfig)
    )
    inference_config: InferenceConfig = attrs.field(
        default=None, converter=to_config(InferenceConfig)
    )
