from typing import List, Tuple

import torch
import torch.nn as nn
from funlib.learn.torch.models import UNet


class UNetModel(nn.Module):  # type: ignore
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 3,
        features_in_last_layer: int = 64,
        downsampling_factors: List[Tuple[int, int]] = [
            (2, 2),
        ],
        num_spatial_dims: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_in_last_layer = features_in_last_layer
        self.backbone = UNet(
            in_channels=self.in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsampling_factors,
            activation="ReLU",
            padding="valid",
            num_fmaps_out=self.features_in_last_layer,
            kernel_size_down=[
                [
                    (3,) * num_spatial_dims,
                    (1,) * num_spatial_dims,
                    (1,) * num_spatial_dims,
                    (3,) * num_spatial_dims,
                ]
            ]
            * (len(downsampling_factors) + 1),
            kernel_size_up=[
                [
                    (3,) * num_spatial_dims,
                    (1,) * num_spatial_dims,
                    (1,) * num_spatial_dims,
                    (3,) * num_spatial_dims,
                ]
            ]
            * len(downsampling_factors),
            constant_upsample=True,
        )
        if num_spatial_dims == 2:
            self.head = torch.nn.Sequential(
                nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer, 1),
                nn.ReLU(),
                nn.Conv2d(self.features_in_last_layer, out_channels, 1),
            )
        elif num_spatial_dims == 3:
            self.head = torch.nn.Sequential(
                nn.Conv3d(self.features_in_last_layer, self.features_in_last_layer, 1),
                nn.ReLU(),
                nn.Conv3d(self.features_in_last_layer, out_channels, 1),
            )

    def head_forward(self, last_layer_output):
        out_cat = self.head(last_layer_output)
        return out_cat

    @staticmethod
    def select_and_add_coordinates(output, coordinates):
        selection = []
        # output.shape = (b, c, h, w)
        for o, c in zip(output, coordinates):
            sel = o[:, c[:, 1], c[:, 0]]
            sel = sel.transpose(1, 0)
            sel += c
            selection.append(sel)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selection, dim=0)

    def forward(self, raw):
        h = self.backbone(raw)
        return self.head_forward(h)
