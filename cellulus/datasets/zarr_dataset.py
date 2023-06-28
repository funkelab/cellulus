import math
from typing import Tuple

import gunpowder as gp
from torch.utils.data import IterableDataset

from cellulus.configs import DatasetConfig

from .meta_data import DatasetMetaData


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
        self.__read_meta_data()

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

    def __read_meta_data(self):
        meta_data = DatasetMetaData(self.dataset_config)

        self.num_dims = meta_data.num_dims
        self.num_spatial_dims = meta_data.num_spatial_dims
        self.num_channels = meta_data.num_channels
        self.sample_dim = meta_data.sample_dim
        self.channel_dim = meta_data.channel_dim
        self.time_dim = meta_data.time_dim

    def get_num_channels(self):
        return self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims
