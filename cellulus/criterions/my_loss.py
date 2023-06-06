import numpy as np
import torch
import torch.nn as nn


class OCELoss(nn.Module):
    def __init__(self, temperature, regularization_weight):
        super().__init__()
        self.temperature = temperature
        self.regularization_weight = regularization_weight
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

    def forward(self, anchor_embedding, reference_embedding):
        dist = self.distance_function(anchor_embedding, reference_embedding.detach())
        nonlinear_dist = self.nonlinearity(dist)
        return nonlinear_dist.sum()
