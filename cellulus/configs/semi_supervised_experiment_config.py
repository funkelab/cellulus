import attrs
from attrs.validators import instance_of

from .inference_config import InferenceConfig
from .model_config import ModelConfig
from .semi_supervised_train_config import SemiSupervisedTrainConfig
from .utils import to_config


@attrs.define
class SemiSupervisedExperimentConfig:
    """Top-level config for a semi-supervised experiment (containing training and prediction).

    Parameters:

        experiment_name:

            A unique name for the experiment.

        object_size:

            A rough estimate of the size of objects in the image, given in
            world units. The "patch size" of the network will be chosen based
            on this estimate.

        model_config:

            The model configuration.

        semi_sup_train_config:

            Configuration object for training the semi-supervised model.

        inference_config:

            Configuration object for prediction.
    """

    experiment_name: str = attrs.field(validator=instance_of(str))
    object_size: float = attrs.field(validator=instance_of(float))

    model_config: ModelConfig = attrs.field(converter=to_config(ModelConfig))
    semi_sup_train_config: SemiSupervisedTrainConfig = attrs.field(
        default=None, converter=to_config(SemiSupervisedTrainConfig)
    )
    inference_config: InferenceConfig = attrs.field(
        default=None, converter=to_config(InferenceConfig)
    )