import numpy as np
import torch
import torch.nn as nn


class OCELoss(nn.Module):  # type: ignore
    def __init__(
        self,
        temperature: float,
        regularization_weight: float,
        density: float,
        kappa: float,
        num_spatial_dims: int,
        reduce_mean: bool,
        device: torch.device,
    ):
        super().__init__()
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.density = density
        self.kappa = kappa
        self.num_spatial_dims = num_spatial_dims
        self.reduce_mean = reduce_mean
        self.device = device

    def distance_function(self, e0, e1):
        diff = e0 - e1
        return diff.norm(2, dim=-1)

    def nonlinearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, prediction):
        if self.num_spatial_dims == 2:
            b, c, h, w = prediction.shape

            num_anchors = int(self.density * h * w)
            anchor_coordinates_y = np.random.randint(
                self.kappa, h - self.kappa, num_anchors
            )
            anchor_coordinates_x = np.random.randint(
                self.kappa, w - self.kappa, num_anchors
            )
            anchor_coordinates = np.stack(
                (anchor_coordinates_x, anchor_coordinates_y), axis=1
            )  # N x 2
        elif self.num_spatial_dims == 3:
            b, c, d, h, w = prediction.shape
            num_anchors = int(self.density * d * h * w)
            anchor_coordinates_z = np.random.randint(
                self.kappa, d - self.kappa, num_anchors
            )
            anchor_coordinates_y = np.random.randint(
                self.kappa, h - self.kappa, num_anchors
            )
            anchor_coordinates_x = np.random.randint(
                self.kappa, w - self.kappa, num_anchors
            )
            anchor_coordinates = np.stack(
                (anchor_coordinates_x, anchor_coordinates_y, anchor_coordinates_z),
                axis=1,
            )  # N x 3
        num_references = int(self.density * np.pi * self.kappa**2)
        anchor_coordinates = np.repeat(anchor_coordinates, num_references, axis=0)
        offsets = self.sample_offsets(
            radius=self.kappa,
            num_samples=len(anchor_coordinates),
        )
        reference_coordinates = anchor_coordinates + offsets
        anchor_coordinates = anchor_coordinates[np.newaxis, ...]
        reference_coordinates = reference_coordinates[np.newaxis, ...]
        anchor_coordinates = torch.from_numpy(np.repeat(anchor_coordinates, b, 0)).to(
            self.device
        )
        reference_coordinates = torch.from_numpy(
            np.repeat(reference_coordinates, b, 0)
        ).to(self.device)
        anchor_embeddings = self.get_embeddings(
            prediction,
            anchor_coordinates,
        )  # B x N x 2/3
        reference_embeddings = self.get_embeddings(
            prediction,
            reference_coordinates,
        )  # B x N x 2/3
        distance = self.distance_function(
            anchor_embeddings, reference_embeddings.detach()
        )
        nonlinear_distance = self.nonlinearity(distance)

        loss = nonlinear_distance + self.regularization_weight * anchor_embeddings.norm(
            2, dim=-1
        )
        if self.reduce_mean:
            return loss.mean()
        else:
            return loss.sum()

    def sample_offsets(self, radius, num_samples):
        if self.num_spatial_dims == 2:
            offset_x = np.random.randint(-radius, radius + 1, size=2 * num_samples)
            offset_y = np.random.randint(-radius, radius + 1, size=2 * num_samples)

            offset_coordinates = np.stack((offset_x, offset_y), axis=1)
        elif self.num_spatial_dims == 3:
            offset_x = np.random.randint(-radius, radius + 1, size=3 * num_samples)
            offset_y = np.random.randint(-radius, radius + 1, size=3 * num_samples)
            offset_z = np.random.randint(-radius, radius + 1, size=3 * num_samples)

            offset_coordinates = np.stack((offset_x, offset_y, offset_z), axis=1)
        in_circle = (offset_coordinates**2).sum(axis=1) < radius**2
        offset_coordinates = offset_coordinates[in_circle]
        not_zero = np.absolute(offset_coordinates).sum(axis=1) > 0
        offset_coordinates = offset_coordinates[not_zero]
        if len(offset_coordinates) < num_samples:
            return self.sample_offsets(radius, num_samples)

        return offset_coordinates[:num_samples]

    def get_embeddings(self, predictions, coordinates):
        selection = []
        for prediction, coordinate in zip(predictions, coordinates):
            if self.num_spatial_dims == 2:
                embedding = prediction[
                    :, coordinate[:, 1].long(), coordinate[:, 0].long()
                ]
            elif self.num_spatial_dims == 3:
                embedding = prediction[
                    :,
                    coordinate[:, 2].long(),
                    coordinate[:, 1].long(),
                    coordinate[:, 0].long(),
                ]
            embedding = embedding.transpose(1, 0)
            embedding += coordinate
            selection.append(embedding)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selection, dim=0)
