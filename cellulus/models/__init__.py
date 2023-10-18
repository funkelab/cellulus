from typing import List, Tuple

from cellulus.models.unet import UNetModel
from cellulus.models.stardist_unet import StarDist_UNetModel


def get_model(
    in_channels: int,
    out_channels: int,
    num_fmaps: int,
    fmap_inc_factor: int,
    features_in_last_layer: int,
    downsampling_factors: List[Tuple[int, ...]],
    num_spatial_dims: int,
    stardist_model: bool = False,
) -> UNetModel:
    if stardist_model:
        return StarDist_UNetModel(
        in_channels=in_channels,
        out_channels=out_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        features_in_last_layer=features_in_last_layer,
        downsampling_factors=downsampling_factors,
        num_spatial_dims=num_spatial_dims,
    )
    return UNetModel(
        in_channels=in_channels,
        out_channels=out_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        features_in_last_layer=features_in_last_layer,
        downsampling_factors=downsampling_factors,
        num_spatial_dims=num_spatial_dims,
    )
