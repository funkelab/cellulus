from typing import Tuple

from cellulus.configs import DatasetConfig
from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(
    dataset_config: DatasetConfig,
    crop_size: Tuple[int, ...],
    control_point_spacing: int,
    control_point_jitter: float,
) -> ZarrDataset:
    return ZarrDataset(
        dataset_config=dataset_config,
        crop_size=crop_size,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
    )
