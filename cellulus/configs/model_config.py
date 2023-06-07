from pathlib import Path
from typing import List

import attrs
from attrs.validators import instance_of

from .utils import to_path


@attrs.define
class ModelConfig:
    """Model configuration.

    Parameters:

        num_fmaps:

            The number of feature maps in the first level of the U-Net.

        fmap_inc_factor:

            The factor by which to increase the number of feature maps between
            levels of the U-Net.

        downsampling_factors:

            A list of downsampling factors, each given per dimension (e.g.,
            [[2,2], [3,3]] would correspond to two downsample layers, one with
            an isotropic factor of 2, and another one with 3). This parameter
            will also determine the number of levels in the U-Net.

        checkpoint (optional, default ``None``):

            A path to a checkpoint of the network. Needs to be set for networks
            that are used for prediction. If set during training, the
            checkpoint will be used to resume training, otherwise the network
            will be trained from scratch.
    """

    num_fmaps: int = attrs.field(validator=instance_of(int))
    fmap_inc_factor: int = attrs.field(validator=instance_of(int))
    downsampling_factors: List[List[int]] = attrs.field(default=[[2, 2]])
    checkpoint: Path = attrs.field(default=None, converter=to_path)
