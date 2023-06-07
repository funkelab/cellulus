import attrs
from attrs.validators import instance_of

from .dataset_config import DatasetConfig
from .utils import to_config


@attrs.define
class TrainConfig:
    """Train configuration.

    Parameters:

        batch_size:

            The number of samples to use per batch.

        max_iterations:

            The maximim number of iterations to train for.

        initial_learning_rate:

            Initial learning rate of the optimizer.

        regularizer_weight:

            The weight of the L2 regularizer on the object-centric embeddings.

        train_data_config:

            Configuration object for the training data.

        validate_data_config:

            Configuration object for the validation data.
    """

    train_data_config: DatasetConfig = attrs.field(converter=to_config(DatasetConfig))
    validate_data_config: DatasetConfig = attrs.field(
        converter=to_config(DatasetConfig)
    )

    batch_size: int = attrs.field(default=8, validator=instance_of(int))
    max_iterations: int = attrs.field(default=100_000, validator=instance_of(int))
    initial_learning_rate: float = attrs.field(
        default=4e-5, validator=instance_of(float)
    )
    regularizer_weight: float = attrs.field(default=1e-5, validator=instance_of(float))
