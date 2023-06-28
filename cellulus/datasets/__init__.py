from pathlib import Path
from typing import Tuple

from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(
    path: Path,
    crop_size: Tuple[int],
    control_point_spacing: int,
    control_point_spacing: float,
) -> ZarrDataset:
    return ZarrDataset(
        path=path,
        crop_size=crop_size,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
    )
