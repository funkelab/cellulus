from pathlib import Path

import attrs
from attrs.validators import instance_of, optional


@attrs.define
class DatasetConfig:
    """Dataset configuration.

    Parameters
    ----------

        container_path:

            A path to the zarr/N5 container.

        dataset_name:

            The name of the dataset containing the raw data in the container.

        secondary_dataset_name:

            The name of the secondary dataset containing the data which needs
            processing.

        'dataset_name' and 'secondary_dataset_name' can be thought of as the
        output and input to a certain task, respectively.
        For example, during segmentation, 'dataset_name' would refer to the output
        segmentation masks and 'secondary_dataset_name' would refer to the input
        predicted embeddings.
        During evaluation, 'dataset_name' would refer to the ground truth masks
        and 'secondary_dataset_name' would refer to the input segmentation masks.

    """

    container_path: Path = attrs.field(converter=Path)
    dataset_name: str = attrs.field(validator=instance_of(str))
    secondary_dataset_name: str = attrs.field(
        default=None, validator=optional(instance_of(str))
    )
