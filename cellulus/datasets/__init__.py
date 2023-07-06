from typing import Tuple

from cellulus.configs import DatasetConfig
from cellulus.datasets.test_zarr_dataset import TestZarrDataset
from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(
    mode: str,
    dataset_config: DatasetConfig,
    crop_size: Tuple[int],
    control_point_spacing: int | None = None,
    control_point_jitter: float | None = None,
) -> ZarrDataset | TestZarrDataset:
    if mode == "train":
        return ZarrDataset(
            dataset_config=dataset_config,
            crop_size=crop_size,
            control_point_spacing=control_point_spacing,
            control_point_jitter=control_point_jitter,
        )
    elif mode == "eval":
        return TestZarrDataset(dataset_config=dataset_config, crop_size=crop_size)
    else:
        raise RuntimeError(f"Mode {mode} not available")
