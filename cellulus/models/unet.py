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
        self.mode = "train"
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

    def set_infer(self, p_salt_pepper, num_infer_iterations):
        self.mode = "infer"
        self.p_salt_pepper = p_salt_pepper
        self.num_infer_iterations = num_infer_iterations

    def head_forward(self, backbone_output):
        out_head = self.head(backbone_output)
        return out_head

    def forward(self, raw):
        if self.mode == "train":
            h = self.backbone(raw)
            return self.head_forward(h)
        elif self.mode == "infer":
            predictions = []
            for val in [0.5, 1.0]:
                for _ in range(self.num_infer_iterations):
                    noisy_input = raw.detach().clone()
                    rnd = torch.rand(*noisy_input.shape).cuda()
                    noisy_input[rnd <= self.p_salt_pepper] = val
                    pred = (
                        self.head_forward(self.backbone(noisy_input))[0].detach().cpu()
                    )
                    predictions.append(pred)

            embedding_std, embedding_mean = torch.std_mean(
                torch.stack(predictions, dim=0), dim=0, keepdim=False, unbiased=False
            )
            embedding_std = embedding_std.sum(dim=0, keepdim=True)
            embedding = torch.cat((embedding_mean, embedding_std), dim=0)
            return torch.unsqueeze(embedding, dim=0)
