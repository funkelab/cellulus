from typing import Tuple

import zarr

from cellulus.configs import DatasetConfig


class DatasetMetaData:
    def __init__(self, shape, axis_names):
        self.num_dims = len(axis_names)
        self.num_spatial_dims: int = 0
        self.num_samples: int = 0
        self.num_channels: int = 0
        self.sample_dim = None
        self.channel_dim = None
        self.time_dim = None
        self.spatial_array: Tuple[int, ...] = ()
        for dim, axis_name in enumerate(axis_names):
            if axis_name == "s":
                self.sample_dim = dim
                self.num_samples = shape[dim]
            elif axis_name == "c":
                self.channel_dim = dim
                self.num_channels = shape[dim]
            elif axis_name == "t":
                self.num_spatial_dims += 1
                self.time_dim = dim
            elif axis_name == "z":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)
            elif axis_name == "y":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)
            elif axis_name == "x":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)

        if self.sample_dim is None:
            self.__invalid_dataset("dataset does not have a sample dimension")

        if self.channel_dim is None:
            self.__invalid_dataset("dataset does not have a channel dimension")

        if self.num_dims != len(shape):
            self.__invalid_dataset(
                f"dataset has {len(shape)} dimensions, but attribute "
                f"axis_names has {self.num_dims} entries"
            )

    @staticmethod
    def from_dataset_config(dataset_config: DatasetConfig) -> "DatasetMetaData":
        container = zarr.open(dataset_config.container_path, "r")
        try:
            data = container[dataset_config.dataset_name]
        except KeyError:
            DatasetMetaData.__invalid_dataset(
                f"Zarr container {dataset_config.container_path} does not contain "
                f'"{dataset_config.dataset_name}" dataset'
            )

        try:
            axis_names = data.attrs["axis_names"]
        except KeyError:
            DatasetMetaData.__invalid_dataset(
                f'"{dataset_config.dataset_name}" dataset in '
                f'{dataset_config.container_path} does not contain "axis_names" '
                "attribute"
            )

        try:
            return DatasetMetaData(data.shape, axis_names)
        except RuntimeError as e:
            raise RuntimeError(
                f'"{dataset_config.dataset_name}" dataset in '
                f"{dataset_config.container_path} has invalid meta-data"
            ) from e

    @staticmethod
    def __invalid_dataset(message):
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
