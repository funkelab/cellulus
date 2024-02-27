import numpy as np
import torch
from sklearn.cluster import MeanShift


def mean_shift_segmentation(
    embedding_mean, embedding_std, bandwidth, min_size, reduction_probability, threshold
):
    embedding_mean = torch.from_numpy(embedding_mean)
    if embedding_mean.ndim == 4:
        embedding_mean[:, 1] += torch.arange(embedding_mean.shape[2])[
            None, :, None
        ]  # += dy
        embedding_mean[:, 0] += torch.arange(embedding_mean.shape[3])[
            None, None, :
        ]  # += dx
    elif embedding_mean.ndim == 5:
        embedding_mean[:, 2] += torch.arange(embedding_mean.shape[2])[
            None, :, None, None
        ]
        embedding_mean[:, 1] += torch.arange(embedding_mean.shape[3])[
            None, None, :, None
        ]
        embedding_mean[:, 0] += torch.arange(embedding_mean.shape[4])[
            None, None, None, :
        ]

    mask = embedding_std < threshold

    mask = mask[None]
    segmentation = segment_with_meanshift(
        embedding_mean,
        bandwidth,
        mask=mask,
        reduction_probability=reduction_probability,
        cluster_all=False,
    )[0]
    return segmentation


def segment_with_meanshift(
    embedding, bandwidth, mask, reduction_probability, cluster_all
):
    anchor_mean_shift = AnchorMeanshift(
        bandwidth, reduction_probability=reduction_probability, cluster_all=cluster_all
    )
    return anchor_mean_shift(embedding, mask=mask) + 1


class AnchorMeanshift:
    def __init__(self, bandwidth, reduction_probability, cluster_all):
        self.mean_shift = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
        self.reduction_probability = reduction_probability

    def compute_mean_shift(self, X):
        if self.reduction_probability < 1.0:
            X_reduced = X[np.random.rand(len(X)) < self.reduction_probability]
            mean_shift_segmentation = self.mean_shift.fit(X_reduced)
        else:
            mean_shift_segmentation = self.mean_shift.fit(X)

        mean_shift_segmentation = self.mean_shift.predict(X)

        return mean_shift_segmentation

    def compute_masked_ms(self, embedding, mask=None):
        if embedding.ndim == 3:
            c, h, w = embedding.shape
            if mask is not None:
                assert len(mask.shape) == 2
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 0).view(w * h, c)
        elif embedding.ndim == 4:
            c, d, h, w = embedding.shape
            if mask is not None:
                assert len(mask.shape) == 3
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 3, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 3, 0).view(d * h * w, c)

        reshaped_embedding = reshaped_embedding.contiguous().numpy()

        mean_shift_segmentation = self.compute_mean_shift(reshaped_embedding)
        if mask is not None:
            mean_shift_segmentation_spatial = -1 * np.ones(mask.shape, dtype=np.int32)
            mean_shift_segmentation_spatial[mask] = mean_shift_segmentation
            mean_shift_segmentation = mean_shift_segmentation_spatial
        else:
            if embedding.ndim == 2:
                mean_shift_segmentation = mean_shift_segmentation.reshape(h, w)
            elif embedding.ndim == 3:
                mean_shift_segmentation = mean_shift_segmentation.reshape(d, h, w)
        return mean_shift_segmentation

    def __call__(self, embedding, mask=None):
        segmentation = []
        for j in range(len(embedding)):
            mask_slice = mask[j] if mask is not None else None
            mean_shift_segmentation = self.compute_masked_ms(
                embedding[j], mask=mask_slice
            )
            segmentation.append(mean_shift_segmentation)

        return np.stack(segmentation)
