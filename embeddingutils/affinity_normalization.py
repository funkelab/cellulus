from .affinities import offset_slice, get_offsets, embedding_to_affinities, \
    label_equal_similarity, ignore_label_mask_similarity
import numpy as np
from scipy.interpolate import interp1d
import torch


def _get_probability_mapping(pred_aff, gt_aff, N=None):
    N = 100 if N is None else N
    f, a_star = pred_aff.flatten(), gt_aff.flatten()
    order = np.argsort(f)
    f, a_star = f[order], a_star[order]
    b = np.array([np.floor(i * len(f) / N) for i in range(N + 1)]).astype(np.int32)
    intervals = [slice(b[i], b[i + 1]) for i in range(N)]
    p = np.array([np.mean(a_star[interval]) for interval in intervals])
    b[-1] -= 1
    c = f[b]
    d = [-np.inf] + [c[i // 2] if i % 2 == 0 else (c[(i - 1) // 2] + c[(i + 1) // 2]) / 2
                     for i in range(2 * N + 1)] + [np.inf]
    p = np.concatenate([[0], p, [1]])  # this assumes affinities to bo small when repulsive

    e = [0, ] + [p[i // 2] if i % 2 == 0 else (p[(i - 1) // 2] + p[(i + 1) // 2]) / 2
                 for i in range(1, 2 * N + 2)] + [1, ]
    return interp1d(d, e)


def get_per_channel_mapping(offsets, pred_aff, gt_aff=None, gt_seg=None, ignore_label=None, return_list_of_mappings=False, **mapping_kwargs):
    """
    given lists of predicted and ground-truth affinities (both in (B, Offsets, (D), H, W) format), returns a mapping
    that can be applied on predicted affinities (in the same format) to make them somehow closer to edge probabilities.
    """
    offsets = get_offsets(offsets)

    if gt_aff is None:
        # derive gt affinities from segmentation
        assert gt_seg is not None
        gt_aff = [np.asarray(embedding_to_affinities(torch.from_numpy(seg), offsets, label_equal_similarity))
                             for seg in gt_seg]

    ignore_mask = None
    if ignore_label is not None:
        assert gt_seg is not None
        assert ignore_label == 0, f'ignore label other than 0 is not supported.'
        # compute ignore masks
        ignore_mask = [np.asarray(embedding_to_affinities(
            torch.from_numpy(seg.astype(np.int32)), offsets, ignore_label_mask_similarity)).astype(np.bool)
            for seg in gt_seg]

    assert len(offsets) == pred_aff[0].shape[0]
    assert all(pred.shape == gt.shape for pred, gt in zip(pred_aff, gt_aff))
    mappings = []
    for i, offset in enumerate(offsets):
        print(f'Computing mapping for offset {offset}')
        flat_pred, flat_gt = np.concatenate(
            [np.stack([pred[i], gt[i]])[(slice(None),) + offset_slice(-offset)].reshape(2, -1)
             for pred, gt in zip(pred_aff, gt_aff)], axis=-1)
        if ignore_mask is not None:
            # mask away meaningless affinities
            mask = np.concatenate([sample_mask[i][offset_slice(-offset)].ravel()
                                   for sample_mask in ignore_mask])
            flat_pred, flat_gt = flat_pred[mask], flat_gt[mask]
        mappings.append(_get_probability_mapping(flat_pred, flat_gt, **mapping_kwargs))

    def mapping(aff):
        assert len(aff.shape) == len(offsets[0]) + 2, f'must be shape (Batch, Offsets, (D), H, W)'
        result = np.ones_like(aff)
        for i, offset in enumerate(offsets):
            result[(slice(None), slice(i, i + 1)) + offset_slice(-offset)] = mappings[i](
                aff[(slice(None), slice(i, i + 1)) + offset_slice(-offset)])
        return result

    if return_list_of_mappings:
        return mapping, mappings
    else:
        return mapping
