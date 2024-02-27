import torch
import torch.nn as nn


class OCELoss(nn.Module):  # type: ignore
    def __init__(
        self,
        temperature: float,
        regularization_weight: float,
        density: float,
        num_spatial_dims: int,
        device: torch.device,
    ):
        """Class definition for loss.

        Parameters
        ----------

            temperature:
                Factor used to scale the gaussian function and control
                the rate of damping.

            regularization_weight:
                The weight of the L2 regularizer on the object-centric embeddings.

            density:
                Determines the fraction of patches to sample per crop,
                during training.

            num_spatial_dims:
                Should be equal to 2 for 2D and 3 for 3D.

            device:
                The device to train on.
                Set to 'cpu' to train without GPU.

        """
        super().__init__()
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.density = density
        self.num_spatial_dims = num_spatial_dims
        self.device = device

    @staticmethod
    def distance_function(embedding_0, embedding_1):
        difference = embedding_0 - embedding_1
        return difference.norm(2, dim=-1)

    def non_linearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, anchor_embedding, reference_embedding):
        distance = self.distance_function(
            anchor_embedding, reference_embedding.detach()
        )
        non_linear_distance = self.non_linearity(distance)
        oce_loss = non_linear_distance.sum()
        regularization_loss = (
            self.regularization_weight * anchor_embedding.norm(2, dim=-1).sum()
        )
        loss = oce_loss + regularization_loss
        return loss, oce_loss, regularization_loss
