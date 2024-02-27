import numpy as np
import torch


class Cluster2d:
    """
    Class for Greedy Clustering of Embeddings on 2D samples.
    """

    def __init__(self, width, height, fg_mask, device):
        """Initializes objects of class `Cluster2d`.

        Parameters
        ----------

            width:

                Width (`W`) of the Raw Image, in number of pixels.

            height:

                Height (`H`) of the Raw Image, in number of pixels.

            fg_mask: (shape is `H` x `W`)

                Foreground Mask corresponding to the region which should be
                partitioned into individual objects.

            device:

                Device on which inference is being run.

        """

        xm = torch.linspace(0, width - 1, width).view(1, 1, -1).expand(1, height, width)
        ym = (
            torch.linspace(0, height - 1, height)
            .view(1, -1, 1)
            .expand(1, height, width)
        )
        xym = torch.cat((xm, ym), 0)
        self.device = device
        self.fg_mask = torch.from_numpy(fg_mask[np.newaxis]).to(self.device)
        self.xym = xym.to(self.device)

    def cluster(
        self,
        prediction,
        bandwidth,
        min_object_size,
        seed_thresh=0.9,
        min_unclustered_sum=0,
    ):
        """Cluster Function.

        Parameters
        ----------

            prediction: (shape is 3 x `H` x `W`)

                Embeddings predicted for the whole raw imnage sample.

            bandwidth:

                Clustering bandwidth or sigma.

            min_object_size:

                Clusters below the `min_object_size` are ignored.

            seed_thresh (default = 0.9):

                Pixels with certainty below 0.9 are ignored to be object
                centers.

            min_unclustered_sum (default = 0):

                If number of pixels which have not been clustered yet falls
                below min_unclustered_sum, the clustering proces stops.


        """
        prediction = torch.from_numpy(prediction).float().to(self.device)
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        embeddings = prediction[0:2] + xym_s  # 2 x h x w
        seed_map = prediction[2:3]  # 1 x h x w
        seed_map_min = seed_map.min()
        seed_map_max = seed_map.max()
        seed_map = (seed_map - seed_map_max) / (seed_map_min - seed_map_max)
        instance_map = torch.zeros(height, width).short()
        count = 1

        embeddings_masked = embeddings[self.fg_mask.expand_as(embeddings)].view(2, -1)
        seed_map_masked = seed_map[self.fg_mask].view(1, -1)
        unclustered = torch.ones(self.fg_mask.sum()).short().to(self.device)
        instance_map_masked = torch.zeros(self.fg_mask.sum()).short().to(self.device)
        while unclustered.sum() > min_unclustered_sum:
            seed = (seed_map_masked * unclustered.float()).argmax().item()
            seed_score = (seed_map_masked * unclustered.float()).max().item()
            if seed_score < seed_thresh:
                break
            center = embeddings_masked[:, seed : seed + 1]
            unclustered[seed] = 0
            dist = torch.exp(
                -1
                * torch.sum(
                    torch.pow(embeddings_masked - center, 2) / (2 * (bandwidth**2)), 0
                )
            )
            proposal = (dist > 0.5).squeeze()
            if proposal.sum() > min_object_size:
                if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                    instance_map_masked[proposal.squeeze()] = count
                    instance_mask = torch.zeros(height, width).short()
                    instance_mask[self.fg_mask.squeeze().cpu()] = proposal.short().cpu()
                    count += 1
            unclustered[proposal] = 0
        instance_map[self.fg_mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map


class Cluster3d:
    """
    Class for Greedy Clustering of Embeddings for 3D samples.
    """

    def __init__(self, width, height, depth, fg_mask, device):
        """Initializes objects of class `Cluster3d`.

        Parameters
        ----------

            width:

                Width (`W`) of the Raw Image, in number of pixels.

            height:

                Height (`H`) of the Raw Image, in number of pixels.

            depth:

                Depth (`D`) of the Raw Image, in number of pixels.

            fg_mask: (shape is `D` x `H` x `W`)

                Foreground Mask corresponding to the region which should be
                partitioned into individual objects.

            device:

                Device on which inference is being run.

        """
        xm = (
            torch.linspace(0, width - 1, width)
            .view(1, 1, 1, -1)
            .expand(1, depth, height, width)
        )
        ym = (
            torch.linspace(0, height - 1, height)
            .view(1, 1, -1, 1)
            .expand(1, depth, height, width)
        )
        zm = (
            torch.linspace(0, depth - 1, depth)
            .view(1, -1, 1, 1)
            .expand(1, depth, height, width)
        )
        xyzm = torch.cat((xm, ym, zm), 0)
        self.device = device
        self.fg_mask = torch.from_numpy(fg_mask[np.newaxis]).to(self.device)
        self.xyzm = xyzm.to(self.device)

    def cluster(
        self,
        prediction,
        bandwidth,
        min_object_size,
        seed_thresh=0.9,
        min_unclustered_sum=0,
    ):
        """Cluster Function..

        Parameters
        ----------

            prediction: (shape is 3 x `D` x `H` x `W`)

                Embeddings predicted for the whole raw imnage sample.

            bandwidth:

                Clustering bandwidth or sigma.

            min_object_size:

                Clusters below the `min_object_size` are ignored.

            seed_thresh (default = 0.9):

                Pixels with certainty below 0.9 are ignored to be object
                centers.

            min_unclustered_sum (default = 0):

                If number of pixels which have not been clustered yet falls
                below min_unclustered_sum, the clustering proces stops.

        """
        prediction = torch.from_numpy(prediction).to(self.device)
        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]
        embeddings = prediction[0:3] + xyzm_s  # 3 x d x h x w
        seed_map = prediction[3:4]  # 1 x d x h x w
        seed_map_min = seed_map.min()
        seed_map_max = seed_map.max()
        seed_map = (seed_map - seed_map_max) / (seed_map_min - seed_map_max)
        instance_map = torch.zeros(depth, height, width).short()
        count = 1

        embeddings_masked = embeddings[self.fg_mask.expand_as(embeddings)].view(3, -1)
        seed_map_masked = seed_map[self.fg_mask].view(1, -1)
        unclustered = torch.ones(self.fg_mask.sum()).short().to(self.device)
        instance_map_masked = torch.zeros(self.fg_mask.sum()).short().to(self.device)
        while unclustered.sum() > min_unclustered_sum:
            seed = (seed_map_masked * unclustered.float()).argmax().item()
            seed_score = (seed_map_masked * unclustered.float()).max().item()
            if seed_score < seed_thresh:
                break
            center = embeddings_masked[:, seed : seed + 1]
            unclustered[seed] = 0
            dist = torch.exp(
                -1
                * torch.sum(
                    torch.pow(embeddings_masked - center, 2) / (2 * (bandwidth**2)), 0
                )
            )
            proposal = (dist > 0.5).squeeze()
            if proposal.sum() > min_object_size:
                if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                    instance_map_masked[proposal.squeeze()] = count
                    instance_mask = torch.zeros(depth, height, width).short()
                    instance_mask[self.fg_mask.squeeze().cpu()] = proposal.short().cpu()
                    count += 1
            unclustered[proposal] = 0
        instance_map[self.fg_mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map
