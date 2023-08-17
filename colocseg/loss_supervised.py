from torch.nn.modules.module import Module
import torch
from torch import Tensor
from torch.nn import functional as F
import gc
from time import sleep

import matplotlib
matplotlib.use('agg')
from matplotlib import collections  as mc
import pylab
import matplotlib.pyplot as plt
import time
from skimage.io import imsave

class SupervisedInstanceEmbeddingLoss(Module):
    def __init__(self, push_margin):
        super().__init__()
        self.push_margin = push_margin

    def pull_distance(self, x, y, dim_channels, dim_samples):
        return (x - y).norm(p=2, dim=dim_channels).mean(dim=dim_samples)

    def push_distance_measure(self, x, y, dim_channels):
        return (self.push_margin - (x-y).norm(p=2, dim=dim_channels)).relu_()

    def push_distance(self, centroids, dim_channels, dim_samples):
        assert centroids.dim() == 2
        distance_matrix = self.push_distance_measure(
            centroids.unsqueeze(dim_samples),
            centroids.unsqueeze(dim_samples+1),
            dim_channels=-1,
        )
        # select vectorized upper triangle of distance matrix
        n_clusters = distance_matrix.shape[0]
        upper_tri_index = torch.arange(1, n_clusters * n_clusters + 1) \
            .view(n_clusters, n_clusters) \
            .triu(diagonal=1).nonzero().transpose(0, 1)
        cluster_distances = distance_matrix[upper_tri_index[0], upper_tri_index[1]]

        return cluster_distances.mean()

    def forward(self, abs_embedding, coordinates, y, split_pull_push=False):
        pull_loss = torch.tensor(0.).to(abs_embedding.device)
        push_loss = torch.tensor(0.).to(abs_embedding.device)

        for b in range(len(y)):
            cx = coordinates[b, :, 1].long()
            cy = coordinates[b, :, 0].long()

            y_per_patch = y[b, cx, cy]
            centroids = []
            dim_channels, dim_samples = 1, 0
            pull_over_instances = 0

            for idx in torch.unique(y_per_patch):
                patch_mask = y_per_patch == idx
                if idx == 0:
                    # skip background instance
                    continue

                instance_embedding = abs_embedding[b, patch_mask]

                centroid = instance_embedding.mean(dim=dim_samples,
                                                   keepdim=True)
                centroids.append(centroid)
                pull_over_instances = pull_over_instances + \
                    self.pull_distance(
                        centroid, instance_embedding, dim_channels, dim_samples)

            # add push loss between centroids
            if len(centroids) > 1:
                pull_loss = pull_loss + (pull_over_instances / len(centroids))
                push_loss = push_loss + self.push_distance(torch.cat(centroids, dim=0),
                                                dim_channels, dim_samples)

        if split_pull_push:
            return pull_loss, push_loss
        else:
            return pull_loss + push_loss
