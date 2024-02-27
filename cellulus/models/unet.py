from typing import List, Tuple

import torch
import torch.nn as nn
from funlib.learn.torch.models import UNet


class UNetModel(nn.Module):  # type: ignore
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_fmaps: int,
        fmap_inc_factor: int,
        features_in_last_layer: int,
        downsampling_factors: List[Tuple[int, ...]],
        num_spatial_dims: int,
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

    def head_forward(self, backbone_output):
        out_head = self.head(backbone_output)
        return out_head

    def forward(self, raw):
        if self.mode == "train":
            h = self.backbone(raw)
            return self.head_forward(h)
        elif self.mode == "infer":
            embeddings = []
            for sample in range(raw.shape[0]):
                raw_sample = raw[sample : sample + 1, ...]
                predictions = []
                for val in [0.5, 1.0]:
                    for _ in range(self.num_infer_iterations):
                        noisy_input = raw_sample.detach().clone()
                        rnd = torch.rand(*noisy_input.shape).to(self.device)
                        noisy_input[rnd <= self.p_salt_pepper] = val
                        prediction = (
                            self.head_forward(self.backbone(noisy_input))[0]
                            .detach()
                            .cpu()
                        )
                        predictions.append(prediction)

                embedding_std, embedding_mean = torch.std_mean(
                    torch.stack(predictions, dim=0),
                    dim=0,
                    keepdim=False,
                    unbiased=False,
                )

                embedding_std = embedding_std.sum(dim=0, keepdim=True)
                embeddings.append(torch.cat((embedding_mean, embedding_std), dim=0))

            return torch.stack(embeddings, dim=0)

    def set_infer(self, p_salt_pepper, num_infer_iterations, device):
        self.mode = "infer"
        self.p_salt_pepper = p_salt_pepper
        self.num_infer_iterations = num_infer_iterations
        self.device: torch.device = device

    @staticmethod
    def select_and_add_coordinates(outputs, coordinates):
        selections = []
        # outputs.shape = (b, c, h, w) or (b, c, d, h, w)
        for output, coordinate in zip(outputs, coordinates):
            if output.ndim == 3:
                selection = output[:, coordinate[:, 1], coordinate[:, 0]]
            elif output.ndim == 4:
                selection = output[
                    :, coordinate[:, 2], coordinate[:, 1], coordinate[:, 0]
                ]
            selection = selection.transpose(1, 0)
            selection += coordinate
            selections.append(selection)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selections, dim=0)
