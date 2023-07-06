import zarr

from cellulus.configs import DatasetConfig


class DatasetMetaData:
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

        container = zarr.open(self.dataset_config.container_path, "r")
        try:
            self.data = container[self.dataset_config.dataset_name]
        except KeyError:
            self.__invalid_dataset(
                f"Zarr container {self.dataset_config.container_path} does not contain "
                f'"{self.dataset_config.dataset_name}" dataset'
            )

        try:
            self.axis_names = self.data.attrs["axis_names"]
        except KeyError:
            self.__invalid_dataset(
                f'"{self.dataset_config.dataset_name}" dataset in '
                f'{self.dataset_config.container_path} does not contain "axis_names" '
                "attribute"
            )

        self.num_dims = len(self.axis_names)
        self.num_spatial_dims = 0
        self.num_samples = None
        self.num_channels = None
        self.sample_dim = None
        self.channel_dim = None
        self.time_dim = None
        self.num_z = None
        self.num_y = None
        self.num_x = None

        for dim, axis_name in enumerate(self.axis_names):
            if axis_name == "s":
                self.sample_dim = dim
                self.num_samples = self.data.shape[dim]
            elif axis_name == "c":
                self.channel_dim = dim
                self.num_channels = self.data.shape[dim]
            elif axis_name == "t":
                self.num_spatial_dims += 1
                self.time_dim = dim
            elif axis_name == "z":
                self.num_spatial_dims += 1
                self.num_z = self.data.shape[dim]
            elif axis_name == "y":
                self.num_spatial_dims += 1
                self.num_y = self.data.shape[dim]
            elif axis_name == "x":
                self.num_spatial_dims += 1
                self.num_x = self.data.shape[dim]

        if self.sample_dim is None:
            self.__invalid_dataset(
                f'"{self.dataset_config.dataset_name}" dataset in '
                f"{self.dataset_config.container_path} does not have a sample dimension"
            )

        if self.channel_dim is None:
            self.__invalid_dataset(
                f'"{self.dataset_config.dataset_name}" dataset in '
                f"{self.dataset_config.container_path} does not have a channel "
                "dimension"
            )
        if self.num_dims != len(self.data.shape):
            self.__invalid_dataset(
                f'"{self.dataset_config.dataset_name}" dataset has '
                f'{len(self.data.shape)} dimensions, but attribute "axis_names" '
                f"has {self.num_dims} entries"
            )

    def __invalid_dataset(self, message):
        raise RuntimeError(
            message
            + "\n\n"
            + (
                "The raw dataset should have shape "
                "(s, c, [t,] [z,] y, x), where s = # of samples, c = # of channels, "
                "t = # of frames, and z/y/x are spatial extents. The dataset should "
                'have an "axis_names" attribute that contains the names of the used '
                'axes, e.g., ["s", "c", "y", "x"] for a 2D dataset.'
            )
        )
