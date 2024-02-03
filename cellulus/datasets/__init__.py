from typing import Tuple

from cellulus.configs import DatasetConfig
from cellulus.datasets.meta_data import DatasetMetaData  # noqa
from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(
    dataset_config: DatasetConfig,
    crop_size: Tuple[int, ...],
    elastic_deform: bool,
    control_point_spacing: int,
    control_point_jitter: float,
) -> ZarrDataset:
    return ZarrDataset(
        dataset_config=dataset_config,
        crop_size=crop_size,
        elastic_deform=elastic_deform,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
    )
