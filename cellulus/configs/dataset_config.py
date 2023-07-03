from pathlib import Path

import attrs
from attrs.validators import instance_of


@attrs.define
class DatasetConfig:
    """Dataset configuration.

    Parameters:

        container_path:

            A path to the zarr/N5 container.

        dataset_name:

            The name of the dataset containing raw data in the container.
    """

    # container_path: Path = attrs.field(converter=Path)
    container_path: str = attrs.field(validator=instance_of(str))
    dataset_name: str = attrs.field(validator=instance_of(str))
