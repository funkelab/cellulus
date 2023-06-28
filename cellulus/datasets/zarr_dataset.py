import math
from typing import Tuple

import gunpowder as gp
import zarr
from torch.utils.data import IterableDataset

from cellulus.configs import DatasetConfig


class ZarrDataset(IterableDataset):  # type: ignore
    def __init__(self, dataset_config: DatasetConfig, crop_size: Tuple[int]):
        """A dataset that serves random samples from a zarr container.

        Args:

            dataset_config:

                A dataset config object pointing to the zarr dataset to use.
                The dataset should have shape `(s, c, [t,] [z,] y, x)`, where
                `s` = # of samples, `c` = # of channels, `t` = # of frames, and
                `z`/`y`/`x` are spatial extents. The dataset should have an
                `"axis_names"` attribute that contains the names of the used
                axes, e.g., `["s", "c", "y", "x"]` for a 2D dataset.


        """

        self.dataset_config = dataset_config
        self.crop_size = crop_size
        self.__open_zarr()

        assert len(crop_size) == self.num_spatial_dims, (
            f'"crop_size" must have the same dimension as the '
            f'spatial(temporal) dimensions of the "{self.dataset_config.dataset_name}" '
            f"dataset which is {self.num_spatial_dims}, but it is {crop_size}"
        )

        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")

        # treat all dimensions as spatial, with a voxel size of 1
        raw_spec = gp.ArraySpec(voxel_size=(1,) * self.num_dims, interpolatable=True)

        spatial_dims = tuple(
            range(self.num_dims - self.num_spatial_dims, self.num_spatial_dims)
        )

        self.pipeline = (
            gp.ZarrSource(
                self.dataset_config.container_path,
                {self.raw: self.dataset_config.dataset_name},
                array_specs={self.raw: raw_spec},
            )
            + gp.RandomLocation()
            + gp.ElasticAugment(
                control_point_spacing=(10,) * self.num_spatial_dims,
                jitter_sigma=(2.0,) * self.num_spatial_dims,
                rotation_interval=(0, math.pi / 2),
                scale_interval=(0.9, 1.1),
                subsample=4,
                spatial_dims=self.num_spatial_dims,
            )
            + gp.SimpleAugment(mirror_only=spatial_dims, transpose_only=spatial_dims)
        )

        # request one sample, all channels, plus crop dimensions
        self.request = gp.BatchRequest()
        self.request[self.raw] = gp.ArraySpec(
            roi=gp.Roi((0,) * self.num_dims, (1, self.num_channels, *self.crop_size))
        )

    def __yield_sample(self):
        """An infinite generator of crops."""

        with gp.build(self.pipeline):
            while True:
                sample = self.pipeline.request_batch(self.request)
                yield sample[self.raw].data[0]

    def __open_zarr(self):
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
            elif axis_name in ["z", "y", "x"]:
                self.num_spatial_dims += 1

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

    def get_num_channels(self):
        return self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims
