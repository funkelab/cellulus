from typing import Tuple

from cellulus.configs import DatasetConfig
from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(
    dataset_config: DatasetConfig,
    crop_size: Tuple[int, ...],
    control_point_spacing: int,
    control_point_jitter: float,
    semi_supervised: bool = False,
    supervised_dataset_config: DatasetConfig = None,
    pseudo_dataset_config: DatasetConfig = None,
) -> ZarrDataset:
    return ZarrDataset(
        dataset_config=dataset_config,
        crop_size=crop_size,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
        semi_supervised = semi_supervised,
        supervised_dataset_config = supervised_dataset_config,
        pseudo_dataset_config = pseudo_dataset_config,
    )
