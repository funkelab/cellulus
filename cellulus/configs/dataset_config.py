from pathlib import Path

import attrs
from attrs.validators import instance_of, optional


@attrs.define
class DatasetConfig:
    """Dataset configuration.

    Parameters:

        container_path:

            A path to the zarr/N5 container.

        dataset_name:

            The name of the dataset containing raw data in the container.

        source_dataset_name:

            The name of the dataset containing the data which needs processing.


    """

    container_path: Path = attrs.field(converter=Path)
    dataset_name: str = attrs.field(validator=instance_of(str))
    source_dataset_name: Path = attrs.field(
        default=None, validator=optional(instance_of(str))
    )
