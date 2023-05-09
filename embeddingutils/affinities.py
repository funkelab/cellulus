import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt


def label_equal_similarity(x, y, dim=0):
    assert x.shape[dim] == 1, 'label images should have one channel only'
    return ((x == y).squeeze(dim=dim)).float()


def ignore_label_mask_similarity(x, y, dim=0, ignore_label=0):
    # returns a mask that is 0 where x or y is to be ignored
    assert x.shape[dim] == 1, 'label images should have one channel only'
    return ((x != ignore_label) * (y != ignore_label)).squeeze(dim=dim)

def label_equal_similarity_with_mask(x, y, dim=0, ignore_label=-1):
    assert x.shape[dim] == 1, 'label images should have one channel only'
    aff = ((x == y).squeeze(dim=dim))
    ignore_mask = (x == ignore_label).add_(y == ignore_label).ge_(1)
    aff[ignore_mask] = ignore_label
    return aff

def label_equal_similarity_with_mask_le(x, y, dim=0, ignore_label_le=-1):
    # this should be a faster implementation in case where all labels smaller 
    # than ignore_label_le should be ignored
    assert x.shape[dim] == 1, 'label images should have one channel only'
    aff = ((x == y)).float()
    
    if ignore_label_le == -1:
        mask = x.min(y).gt_(ignore_label_le).float()
        aff.add_(1).mul_(mask).add_(-1)
    elif ignore_label_le == 0:
        mask = x.min(y).gt_(ignore_label_le).float()
        aff.mul_(mask)
    else:
        mask = x.min(y).le_(ignore_label_le)
        aff[mask] = -1
    return aff.squeeze(dim=dim)


def label_equal_similarity_with_mask_max_le(x, y, dim=0, ignore_label_le=-1):
    # This is a specialized implementation that masks only edges
    # where both incident nodes have a label less or equal to ignore_label_le
    assert x.shape[dim] == 1, 'label images should have one channel only'
    aff = ((x == y)).float()
    
    if ignore_label_le == -1:
        mask = x.max(y).gt_(ignore_label_le).float()
        aff.add_(1).mul_(mask).add_(-1)
    elif ignore_label_le == 0:
        mask = x.max(y).gt_(ignore_label_le).float()
        aff.mul_(mask)
    else:
        mask = x.max(y).le_(ignore_label_le)
        aff[mask] = ignore_label_le
    return aff.squeeze(dim=dim)

def euclidean_distance(x, y, dim=0):
    return (x - y).norm(p=2, dim=dim)


def squared_euclidean_distance(x, y, dim=0):
    return ((x - y) ** 2).sum(dim)


def l1_distance(x, y, dim=0):
    return (x - y).norm(p=1, dim=dim)


def mean_l1_distance(x, y, dim=0):
    # l1 distance divided by dimension. was previously used in segmentwise free loss
    return l1_distance(x, y, dim=dim) / x.size(dim)


def logistic_similarity(x, y, dim=0, offset=None):
    if offset is None:
        sig = 0.1
    else:
        sig = np.linalg.norm(offset)
    return 2 / (1 + (1 / torch.exp(-squared_euclidean_distance(x, y, dim=dim)/(2 * sig**2)).clamp(min=1e-10)))


def cosine_distance(x, y, dim=0):
    return 0.5 * (1 - F.cosine_similarity(x, y, dim=dim))


def normalized_cosine_similarity(x, y, dim=0):
    return 0.5 * (1 + F.cosine_similarity(x, y, dim=dim))


def normalized_cosine_similarity_margin1(x, y, dim=0):
    return F.relu(normalized_cosine_similarity(x, y, dim=dim) * 2 - 1)


def offset_slice(offset, reverse=False, extra_dims=0):
    def shift(o):
        if o == 0:
            return slice(None)
        elif o > 0:
            return slice(o, None)
        else:
            return slice(0, o)
    if not reverse:
        return (slice(None),) * extra_dims + tuple(shift(int(o)) for o in offset)
    else:
        return (slice(None),) * extra_dims + tuple(shift(-int(o)) for o in offset)


def offset_padding(offset):
    result = []
    for o in reversed(offset):
        result.append(int(max(-o, 0)))
        result.append(int(max(o, 0)))
    return tuple(result)


def get_offsets(offsets):
    if isinstance(offsets, str):
        if offsets == 'default-3D':
            offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],
                                [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                                [-1, -1, 0], [-1, 0, -1],
                                [0, -27, 0], [0, 0, -27]], int)
        elif offsets == 'long-3D':
            offsets = np.array([[-1, 0, 0],
                                [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],
                                [0, -9, 4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                                [-1, -9, 0], [-1, 0, -9],  # this is changed, too!
                                [0, -27, 0], [0, 0, -27]], int)
        elif offsets == 'minimal-3D':
            offsets = np.array([[-1, 0, 0], [0, -1, 0],
                                [0, -9, 0],
                                [0, -27, 0]], int)
        elif offsets == 'default-2D':
            offsets = np.array([[-1, 0], [0, -1],
                                [-9, 0], [0, -9],
                                [-9, -9], [9, -9],
                                [-9, -4], [-4, -9], [4, -9], [9, -4],
                                [-27, 0], [0, -27]], int)
        elif offsets == 'long-2D':
            offsets = np.array([[-9, 0], [0, -9],
                                [-9, -9], [9, -9],
                                [-9, -4], [-4, -9], [4, -9], [9, -4],
                                [-27, 0], [0, -27]], int)
        elif offsets == 'tracking-long-2D':
            offsets = np.array([[-1, 0, 0], [-2, 0, 0],
                                [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],
                                [-1, -9, 4], [-1, -4, -9], [-1, 4, -9], [-1, 9, -4],
                                [0, -9, 0], [0, 0, -9],
                                [0, -9, -9], [0, 9, -9],
                                [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                                [0, -27, 0], [0, 0, -27]], int)
        elif offsets == 'default-2D-in-3D':
            offsets = get_offsets('default-2D')
            offsets = np.concatenate([np.zeros((len(offsets), 1)), offsets], axis=-1)
        elif offsets == 'long-2D-in-3D':
            offsets = get_offsets('long-2D')
            offsets = np.concatenate([np.zeros((len(offsets), 1)), offsets], axis=-1)
        else:
            assert False, "Please provide a list of offsets or one of " \
                          "['default-3D', 'long-3D', 'minimal-3D', 'default-2D', 'default-2D-in-3D']"
    return offsets if isinstance(offsets, np.ndarray) else np.array(offsets, int)


def embedding_to_affinities(emb, offsets='default-3D', affinity_measure=euclidean_distance,
                            pass_offset=False, pad_val=1.):
    # if len(emb.shape) = n + 1 + len(offsets[0]):
    # function is parallel over first n dimensions
    # the (n+1)th dimension is assumed to be embedding dimenstion
    # rest are going to be shifted by offsets
    if affinity_measure is None:
        return emb

    offsets = get_offsets(offsets)

    result = []
    emb_axis = len(emb.shape) - len(offsets[0]) - 1
    extra_dims = len(emb.shape) - len(offsets[0])
    for off in offsets:
        if all(abs(o) < s for o, s in zip(off, emb.shape[-len(off):])):
            s1 = offset_slice(off, reverse=True, extra_dims=extra_dims)
            s2 = offset_slice(off, extra_dims=extra_dims)
            if not pass_offset:
                aff = affinity_measure(emb[s1], emb[s2], dim=emb_axis)
            else:
                aff = affinity_measure(emb[s1], emb[s2], dim=emb_axis, offset=off)
            aff = F.pad(aff, offset_padding(off), value=pad_val)
        else:
            print('warning: offset bigger than image')
            aff = torch.zeros(emb.shape[:emb_axis] + emb.shape[emb_axis+1:]).to(emb.device)
        result.append(aff)

    return torch.stack(result, dim=emb_axis)


class EmbeddingToAffinities(torch.nn.Module):
    def __init__(self, offsets='default-3D', affinity_measure=logistic_similarity, pass_offset=False):
        super(EmbeddingToAffinities, self).__init__()
        self.offsets = get_offsets(offsets)
        self.affinity_measure = affinity_measure
        self.pass_offset = pass_offset
        # TODO: compute offset slices and paddings in constructor

    def forward(self, emb):
        if self.pass_offset:
            return embedding_to_affinities(emb,
                                           offsets=self.offsets,
                                           affinity_measure=self.affinity_measure,
                                           pass_offset=self.pass_offset)
        else:
            return embedding_to_affinities(emb,
                                           offsets=self.offsets,
                                           affinity_measure=self.affinity_measure)


def seg_to_borders(seg, dim=2, thickness=1):
    offsets = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    offsets = np.concatenate([offsets * i for i in range(1, thickness + 1)])
    shape = seg.shape
    seg = seg.reshape((-1,) + shape[-dim:])
    result = []
    for s in seg:
        result.append(1 - embedding_to_affinities(
            s.unsqueeze(-dim-1),
            offsets=offsets,
            affinity_measure=label_equal_similarity
        ).min(-dim-1, keepdim=True)[0].squeeze(-dim-1))
    return torch.stack(result).contiguous().view(shape)


if __name__ == '__main__':
    t = torch.arange(25).view(5, 5)
    print(t)
    off = (2, 0)
    pad = offset_padding(off)
    print(pad)
    s = offset_slice(off)
    print(s)
    print(t[s])
    print(F.pad(t[s], pad))
    print('-'*100)
    emb = torch.eye(10).view(1, 10, 10)
    print(emb)
    offsets = ((1, -5), (1, -1))
    print(embedding_to_affinities(emb, offsets))
