from pathlib import Path
from typing import List

import attrs
from attrs.validators import instance_of

from .utils import to_path


@attrs.define
class ModelConfig:
    """Model Configuration.

    Parameters
    ----------

        num_fmaps:

            The number of feature maps in the first level of the U-Net.

        fmap_inc_factor:

            The factor by which to increase the number of feature maps between
            levels of the U-Net.

        features_in_last_layer (default = 64):

            The number of feature channels in the last layer of the U-Net

        downsampling_factors (default = [[2,2]]):

            A list of downsampling factors, each given per dimension (e.g.,
            [[2,2], [3,3]] would correspond to two downsample layers, one with
            an isotropic factor of 2, and another one with 3). This parameter
            will also determine the number of levels in the U-Net.

        checkpoint (default = None):

            A path to a checkpoint of the network. Needs to be set for networks
            that are used for prediction. If set during training, the
            checkpoint will be used to resume training, otherwise the network
            will be trained from scratch.

        initialize (default = True)

            If True, initialize the model weights with Kaiming Normal.

    """

    num_fmaps: int = attrs.field(validator=instance_of(int))
    fmap_inc_factor: int = attrs.field(validator=instance_of(int))
    features_in_last_layer: int = attrs.field(default=64)
    downsampling_factors: List[List[int]] = attrs.field(
        default=[
            [2, 2],
        ]
    )
    checkpoint: Path = attrs.field(default=None, converter=to_path)
    initialize: bool = attrs.field(default=True, validator=instance_of(bool))
