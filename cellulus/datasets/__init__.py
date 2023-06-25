from pathlib import Path
from typing import Tuple

from cellulus.datasets.zarr_dataset import ZarrDataset


def get_dataset(path: Path, crop_size: Tuple[int]) -> ZarrDataset:
    return ZarrDataset(path=path, crop_size=crop_size)
