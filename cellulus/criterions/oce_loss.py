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
    ):
        super().__init__()
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.density = density
        self.kappa = kappa

        print(
            "Created OCE Loss-object with temperature={} and regularization={}".format(
                temperature, regularization_weight
            )
        )

    def distance_function(self, e0, e1):
        diff = e0 - e1
        return diff.norm(2, dim=-1)

    def nonlinearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, prediction):
        if prediction.dim() == 4:
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
        elif prediction.dim() == 5:
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
            ndim=prediction.dim(),
        )
        reference_coordinates = anchor_coordinates + offsets
        reference_coordinates = reference_coordinates.astype(int)
        anchor_coordinates = anchor_coordinates[np.newaxis, ...]
        reference_coordinates = reference_coordinates[np.newaxis, ...]
        anchor_coordinates = torch.from_numpy(np.repeat(anchor_coordinates, b, 0))
        reference_coordinates = torch.from_numpy(np.repeat(reference_coordinates, b, 0))
        anchor_embeddings = self.get_embeddings(
            prediction, anchor_coordinates, prediction.dim()
        )  # B x N x 2/3
        reference_embeddings = self.get_embeddings(
            prediction, reference_coordinates, prediction.dim()
        )  # B x N x 2/3
        distance = self.distance_function(
            anchor_embeddings, reference_embeddings.detach()
        )
        nonlinear_distance = self.nonlinearity(distance)

        return (
            nonlinear_distance.sum()
            + self.regularization_weight * anchor_embeddings.norm(2, dim=-1).sum()
        )

    def sample_offsets(self, radius, num_samples, ndim):
        if ndim == 4:
            theta = 2 * np.pi * np.random.random(num_samples)
            r = radius * np.random.random(num_samples)
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            offsets = np.stack((dx, dy), axis=1)
        elif ndim == 5:
            theta = 2 * np.pi * np.random.random(num_samples)
            r = radius * np.random.random(num_samples)
            phi = np.pi * np.random.random(num_samples)
            dz = r * np.cos(phi)
            dy = r * np.sin(phi) * np.sin(theta)
            dx = r * np.sin(phi) * np.cos(theta)
            offsets = np.stack((dx, dy, dz), axis=1)
        return offsets

    def get_embeddings(self, predictions, coordinates, dim):
        selection = []
        for prediction, coordinate in zip(predictions, coordinates):
            if dim == 4:
                embedding = prediction[:, coordinate[:, 1], coordinate[:, 0]]
            elif dim == 5:
                embedding = prediction[
                    :, coordinate[:, 2], coordinate[:, 1], coordinate[:, 0]
                ]
            embedding = embedding.transpose(1, 0)
            embedding += coordinate
            selection.append(embedding)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selection, dim=0)
