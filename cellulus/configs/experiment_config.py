from datetime import datetime

import attrs
from attrs.validators import instance_of

from .inference_config import InferenceConfig
from .model_config import ModelConfig
from .train_config import TrainConfig
from .utils import to_config


@attrs.define
class ExperimentConfig:
    """Top-level config for an experiment (containing training and prediction).

    Parameters
    ----------

        experiment_name: (default = 'YYYY-MM-DD')

            A unique name for the experiment.

        object_size: (default = 30)

            A rough estimate of the size of objects in the image, given in
            world units. The "patch size" of the network will be chosen based
            on this estimate.

        normalization_factor: (default = None)

            The factor to use, for dividing the raw image pixel intensities.
            If 'None', a factor is chosen based on the dtype of the array .
            (e.g., np.uint8 would result in a factor of 1.0/255).

        model_config:

            Configuration object for the model.

        train_config:

            Configuration object for training.

        inference_config:

            Configuration object for prediction.
    """

    model_config: ModelConfig = attrs.field(converter=to_config(ModelConfig))
    experiment_name: str = attrs.field(
        default=datetime.today().strftime("%Y-%m-%d"), validator=instance_of(str)
    )
    normalization_factor: float = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(float))
    )
    object_size: int = attrs.field(default=30, validator=instance_of(int))

    train_config: TrainConfig = attrs.field(
        default=None, converter=to_config(TrainConfig)
    )
    inference_config: InferenceConfig = attrs.field(
        default=None, converter=to_config(InferenceConfig)
    )
