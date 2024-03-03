import math
from typing import Tuple

import gunpowder as gp
import numpy as np
from torch.utils.data import IterableDataset

from cellulus.configs import DatasetConfig

from .meta_data import DatasetMetaData


class ZarrDataset(IterableDataset):  # type: ignore
    def __init__(
        self,
        dataset_config: DatasetConfig,
        crop_size: Tuple[int, ...],
        elastic_deform: bool,
        control_point_spacing: int,
        control_point_jitter: float,
        density: float,
        kappa: float,
        normalization_factor: float,
    ):
        """A dataset that serves random samples from a zarr container.

        Args:

            dataset_config:

                A dataset config object pointing to the zarr dataset to use.
                The dataset should have shape `(s, c, [t,] [z,] y, x)`, where
                `s` = # of samples, `c` = # of channels, `t` = # of frames, and
                `z`/`y`/`x` are spatial extents. The dataset should have an
                `"axis_names"` attribute that contains the names of the used
                axes, e.g., `["s", "c", "y", "x"]` for a 2D dataset.


            crop_size:

                The size of data crops used during training (distinct from the
                "patch size" of the method: from each crop, multiple patches
                will be randomly selected and the loss computed on them). This
                should be equal to the input size of the model that predicts
                the OCEs.

            elastic_deform:

                Whether to elastically deform data in order to augment training samples?

            control_point_spacing:

                The distance in pixels between control points used for elastic
                deformation of the raw data.
                Only used, if `elastic_deform` is set to True.

            control_point_jitter:

                How much to jitter the control points for elastic deformation
                of the raw data, given as the standard deviation of a normal
                distribution with zero mean.
                Only used if `elastic_deform` is set to True.

            density:

                Determines the fraction of patches to sample per crop, during training.

            kappa:

                Neighborhood radius to extract patches from.

            normalization_factor:

                The factor to use, for dividing the raw image pixel intensities.
                If 'None', a factor is chosen based on the dtype of the array .
                (e.g., np.uint8 would result in a factor of 1.0/255).
        """

        self.dataset_config = dataset_config
        self.crop_size = crop_size
        self.elastic_deform = elastic_deform
        self.control_point_spacing = control_point_spacing
        self.control_point_jitter = control_point_jitter
        self.normalization_factor = normalization_factor
        self.__read_meta_data()

        assert len(crop_size) == self.num_spatial_dims, (
            f'"crop_size" must have the same dimension as the '
            f'spatial(temporal) dimensions of the "{self.dataset_config.dataset_name}" '
            f"dataset which is {self.num_spatial_dims}, but it is {crop_size}"
        )
        self.density = density
        self.kappa = kappa
        self.output_shape = tuple(int(_ - 16) for _ in self.crop_size)
        self.normalization_factor = normalization_factor
        self.unbiased_shape = tuple(
            int(_ - (2 * self.kappa)) for _ in self.output_shape
        )
        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")

        # treat all dimensions as spatial, with a voxel size of 1
        raw_spec = gp.ArraySpec(voxel_size=(1,) * self.num_dims, interpolatable=True)

        # spatial_dims = tuple(range(self.num_dims - self.num_spatial_dims,
        # self.num_dims))

        self.pipeline = (
            gp.ZarrSource(
                self.dataset_config.container_path,
                {self.raw: self.dataset_config.dataset_name},
                array_specs={self.raw: raw_spec},
            )
            + gp.RandomLocation()
            + gp.Normalize(self.raw, factor=self.normalization_factor)
        )

        if self.elastic_deform:
            self.pipeline += gp.ElasticAugment(
                control_point_spacing=(self.control_point_spacing,)
                * self.num_spatial_dims,
                jitter_sigma=(self.control_point_jitter,) * self.num_spatial_dims,
                rotation_interval=(0, math.pi / 2),
                scale_interval=(0.9, 1.1),
                subsample=4,
                spatial_dims=self.num_spatial_dims,
            )
            # + gp.SimpleAugment(mirror_only=spatial_dims, transpose_only=spatial_dims)

    def __yield_sample(self):
        """An infinite generator of crops."""

        with gp.build(self.pipeline):
            while True:
                array_is_zero = True
                # request one sample, all channels, plus crop dimensions
                while array_is_zero:
                    request = gp.BatchRequest()
                    request[self.raw] = gp.ArraySpec(
                        roi=gp.Roi(
                            (0,) * self.num_dims,
                            (1, self.num_channels, *self.crop_size),
                        )
                    )

                    sample = self.pipeline.request_batch(request)
                    sample_data = sample[self.raw].data[0]
                    if np.max(sample_data) <= 0.0:
                        pass
                    else:
                        array_is_zero = False
                        anchor_samples, reference_samples = self.sample_coordinates()
                yield sample_data, anchor_samples, reference_samples

    def __read_meta_data(self):
        meta_data = DatasetMetaData.from_dataset_config(self.dataset_config)

        self.num_dims = meta_data.num_dims
        self.num_spatial_dims = meta_data.num_spatial_dims
        self.num_channels = meta_data.num_channels
        self.num_samples = meta_data.num_samples
        self.sample_dim = meta_data.sample_dim
        self.channel_dim = meta_data.channel_dim
        self.time_dim = meta_data.time_dim

    def get_num_channels(self):
        return self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims

    def sample_offsets_within_radius(self, radius, number_offsets):
        if self.num_spatial_dims == 2:
            offsets_x = np.random.randint(-radius, radius + 1, size=2 * number_offsets)
            offsets_y = np.random.randint(-radius, radius + 1, size=2 * number_offsets)
            offsets_coordinates = np.stack((offsets_x, offsets_y), axis=1)
        elif self.num_spatial_dims == 3:
            offsets_x = np.random.randint(-radius, radius + 1, size=3 * number_offsets)
            offsets_y = np.random.randint(-radius, radius + 1, size=3 * number_offsets)
            offsets_z = np.random.randint(-radius, radius + 1, size=3 * number_offsets)
            offsets_coordinates = np.stack((offsets_x, offsets_y, offsets_z), axis=1)

        in_circle = (offsets_coordinates**2).sum(axis=1) < radius**2
        offsets_coordinates = offsets_coordinates[in_circle]
        not_zero = np.absolute(offsets_coordinates).sum(axis=1) > 0
        offsets_coordinates = offsets_coordinates[not_zero]

        if len(offsets_coordinates) < number_offsets:
            return self.sample_offsets_within_radius(radius, number_offsets)

        return offsets_coordinates[:number_offsets]

    def sample_coordinates(self):
        num_anchors = self.get_num_anchors()
        num_references = self.get_num_references()

        if self.num_spatial_dims == 2:
            anchor_coordinates_x = np.random.randint(
                self.kappa,
                self.output_shape[0] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_y = np.random.randint(
                self.kappa,
                self.output_shape[1] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates = np.stack(
                (anchor_coordinates_x, anchor_coordinates_y), axis=1
            )
        elif self.num_spatial_dims == 3:
            anchor_coordinates_x = np.random.randint(
                self.kappa,
                self.output_shape[0] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_y = np.random.randint(
                self.kappa,
                self.output_shape[1] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_z = np.random.randint(
                self.kappa,
                self.output_shape[2] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates = np.stack(
                (anchor_coordinates_x, anchor_coordinates_y, anchor_coordinates_z),
                axis=1,
            )
        anchor_samples = np.repeat(anchor_coordinates, num_references, axis=0)
        offset_in_pos_radius = self.sample_offsets_within_radius(
            self.kappa, len(anchor_samples)
        )
        reference_samples = anchor_samples + offset_in_pos_radius

        return anchor_samples, reference_samples

    def get_num_anchors(self):
        return int(self.density * self.unbiased_shape[0] * self.unbiased_shape[1])

    def get_num_references(self):
        return int(self.density * self.kappa**2 * np.pi)

    def get_num_samples(self):
        return self.get_num_anchors() * self.get_num_references()
