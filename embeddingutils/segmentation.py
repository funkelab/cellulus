import torch
from torch.nn.functional import cosine_similarity
from skimage.measure import label
import numpy as np
import collections
from embeddingutils.affinities import embedding_to_affinities, get_offsets, logistic_similarity

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from affogato.segmentation import compute_mws_clustering
except ImportError:
    compute_mws_clustering = None

try:
    import constrained_mst as cmst
except ImportError:
    constrained_mst = None


def mws_segmentation(embedding, offsets='default-3D', affinity_measure=logistic_similarity, pass_offset=False,
                     ATT_C=None, repulsive_strides=None, percentile=5, return_affinities=False,
                     attraction_factor=1, z_delay=0):
    assert cmst is not None, 'need constrained_mst for mws_segmentation'

    offsets = get_offsets(offsets)
    if ATT_C is None:
        ATT_C = len(offsets[0])

    emb_shape = embedding.shape
    img_shape = embedding.shape[-len(offsets[0]):]
    n_img_dims = len(img_shape)

    if repulsive_strides is None:
        repulsive_strides = (1,) * (n_img_dims - 2) + (8, 8)
    repulsive_strides = np.array(repulsive_strides, dtype=int)

    # actually not needed, huge offsets are fine
    # for off in offsets:
    #    assert all(abs(o) < s for o, s in zip(off, emb.shape[-len(off):])), \
    #        f'offset {off} is to big for image of shape {img_shape}'
    if affinity_measure is not None:
        affinities = embedding_to_affinities(embedding, offsets=offsets, affinity_measure=affinity_measure,
                                             pass_offset=pass_offset)
    else:
        affinities = embedding
    affinities = affinities.contiguous().view((-1, len(offsets)) + emb_shape[-n_img_dims:])

    if percentile is not None:
        affinities -= np.percentile(affinities, percentile)
        affinities[:, :ATT_C] *= -1
    else:
        affinities[:, :ATT_C] *= -1
        affinities[:, :ATT_C] += 1
    affinities[:, :ATT_C] *= attraction_factor
    if z_delay != 0:
        affinities[:, (offsets[:, 0] != 0).astype(np.uint8)] += z_delay

    result = []
    for aff in affinities:
        dws = cmst.ConstrainedWatershed(np.array(img_shape),
                                        offsets,
                                        ATT_C,
                                        repulsive_strides)
        sorted_edges = np.argsort(aff, axis=None)
        dws.repulsive_ucc_mst_cut(sorted_edges, 0)
        seg = label(dws.get_flat_label_image().reshape(img_shape))
        seg = np.random.permutation(seg.max() + 1)[seg]
        result.append(seg)

    result = np.stack(result, axis=-(n_img_dims + 1)).reshape(emb_shape[:-n_img_dims - 1] + emb_shape[-n_img_dims:])

    if return_affinities:
        return torch.from_numpy(result), affinities
    else:
        return torch.from_numpy(result)


def nonlocal_mws_segmentation(att_aff, offsets,
                              mutex_edges, mutex_edge_weights):
    assert compute_mws_clustering is not None, 'need compute_mws_clustering for nonlocal_mws_segmentation'

    # determine the strides of the image
    shape = att_aff.shape[1:]
    array_stride = [0] * len(shape)
    current_stride = 1
    for i in reversed(range(len(shape))):
        array_stride[i] = current_stride
        current_stride *= shape[i]

    # generate attractive edge list
    attractive_edge_weights = []
    attractive_edges = []

    # check which edges leave image
    valid_edges = np.ones(att_aff.shape, dtype=bool)
    ndim = len(offsets[0])
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            # invalid_slice = (i, ) + i * (slice(None), ) + slice(o)
            inv_slice = slice(0, -o) if o < 0 else slice(shape[j] - o, shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0

    # generate list of valid attractive edges by (u, v)
    source_points = np.arange(np.product(shape)).reshape(shape)
    for idx, offset in enumerate(offsets):

        # compute stride
        stride = 0
        for i in range(len(offset)):
            stride += offset[i] * array_stride[i]

        target_points = source_points + stride
        attractive_edges.append(np.stack([source_points[valid_edges[idx]].ravel(),
                                          target_points[valid_edges[idx]].ravel()], axis=1))
        attractive_edge_weights.append(att_aff[idx][[valid_edges[idx]]].ravel())

    attractive_edges = np.concatenate(attractive_edges, axis=0)
    attractive_edge_weights = np.concatenate(attractive_edge_weights, axis=0)

    number_of_labels = np.product(shape)
    node_labels = compute_mws_clustering(number_of_labels,
                                         attractive_edges.astype('uint64'),
                                         mutex_edges.astype('uint64'),
                                         (1 - attractive_edge_weights).astype('float'),
                                         (1 - mutex_edge_weights).astype('float'))

    return node_labels.reshape(shape)


def _append_coords(embedding, coord_scales):
    # embedding should be torch tensor
    n_img_dims = len(coord_scales)
    emb_shape = embedding.shape
    img_shape = emb_shape[-n_img_dims:]
    n_pixels = np.product(img_shape)

    # reshape embedding
    embedding = embedding.contiguous().view(-1, emb_shape[-n_img_dims - 1], n_pixels).permute(0, 2, 1)

    coord_axes = []
    for i, scale in enumerate(coord_scales):
        coord_axes.append(np.linspace(0, (img_shape[i] - 1) * scale, img_shape[i], dtype=np.float32))
    coord_mesh = np.stack(np.meshgrid(*coord_axes), axis=-1).reshape(n_pixels, -1)[None].repeat(embedding.shape[0], 0)
    embedding = torch.cat([torch.from_numpy(coord_mesh).type(embedding.dtype), embedding], dim=-1)

    # restore shape
    resulting_shape = list(emb_shape)
    resulting_shape[-(n_img_dims + 1)] += len(coord_scales)
    return embedding.permute(0, 2, 1).reshape(resulting_shape)


def hdbscan_segmentation(embedding, n_img_dims=None, coord_scales=None,
                         metric='euclidean', min_cluster_size=50, slice_for_fit=None, **hdbscan_kwargs):
    assert hdbscan is not None, 'need hdbscan for hdbscan_segmentation'
    assert metric in hdbscan.dist_metrics.METRIC_MAPPING
    if n_img_dims is None:
        # default: assume one embedding image is being passed
        n_img_dims = len(embedding.shape) - 1
    emb_shape = embedding.shape
    img_shape = emb_shape[-n_img_dims:]

    # append image coordinates as features if requested
    if coord_scales is not None:
        if not isinstance(coord_scales, collections.Iterable):
            coord_scales = n_img_dims * (coord_scales,)
        assert len(coord_scales) == n_img_dims, f'{coord_scales}, {n_img_dims}'
        embedding = _append_coords(embedding, coord_scales)

    # compute #pixels per image
    n_pixels = 1
    for s in img_shape:
        n_pixels *= s

    # reshape embedding for clustering
    embedding = embedding.contiguous().view(-1, embedding.shape[-n_img_dims - 1], n_pixels).permute(0, 2, 1)

    # init HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, **hdbscan_kwargs)

    # iterate over images in batch
    result = []
    for emb in embedding:
        if slice_for_fit is not None:
            clusterer.fit(emb.view(*img_shape, -1)[slice_for_fit].contiguous().view(-1, emb.shape[-1]))
            clusterer.generate_prediction_data()
            labels = hdbscan.approximate_predict(clusterer, emb).reshape(img_shape)
        else:
            labels = clusterer.fit_predict(emb).reshape(img_shape)
        result.append(labels)

    result = np.stack(result, axis=0).reshape(emb_shape[:-n_img_dims - 1] + emb_shape[-n_img_dims:])
    return torch.from_numpy(result)


def gaussian_mask_growing_segmentation(embedding, seed_map, sigma_map, seed_score_threshold=0.5, min_cluster_size=1):
    """
    Compute a segmentation by sequentially growing masks based on a gaussian similarity.
    :param embedding: torch.FloatTensor
    Shape should be (E D H W) or (E H W) (embedding dimension + spatial dimensions).
    :param seed_map: torch.FloatTensor
    Shape should be (D H W) or (H W) (arbitrary number of spatial dimensions).
    :param sigma_map: torch.FloatTensor
    Shape should be (D H W) or (H W) (arbitrary number of spatial dimensions).
    :param seed_score_threshold float
    If no seeds with at least this score are present, clustering stops.
    :param min_cluster_size int
    Clulsters with a smaller number of pixels will be discarded.
    :return: torch.LongTensor
    the computed segmentation
    """
    # TODO: argmax over masks as final step
    spatial_shape = embedding.shape[1:]
    segmentation = torch.zeros(spatial_shape).long().to(embedding.device)
    mask = seed_map > 0.5

    if mask.sum() < min_cluster_size:  # nothing to do
        return segmentation

    masked_embedding = embedding[:, mask]
    masked_seed_map = seed_map[mask]
    masked_sigma_map = sigma_map[mask]
    masked_segmentation = torch.zeros_like(masked_seed_map).long()
    unclustered = torch.ones_like(masked_segmentation).bool()

    current_id = 1
    while unclustered.sum() >= min_cluster_size:
        # get position of seed with highest score
        seed = (masked_seed_map * unclustered.float()).argmax().item()
        seed_score = masked_seed_map[seed]

        if seed_score < seed_score_threshold:
            break

        center = masked_embedding[:, seed]
        unclustered[seed] = 0  # just to make sure
        sigma = masked_sigma_map[seed]
        smooth_mask = torch.exp(-1 * ((((masked_embedding - center[:, None]) / sigma) ** 2).sum(0)))

        proposal = smooth_mask > 0.5

        if proposal.sum() > min_cluster_size:
            if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:  # half of proposal not assigned yet
                masked_segmentation[proposal] = current_id
                current_id += 1

        unclustered[proposal] = 0

    segmentation[mask] = masked_segmentation

    return segmentation


if __name__ == '__main__':

    import h5py
    with h5py.File("/export/home/swolf/local/src/data/skeleton_affinities.h5") as skaff:
        att_aff = skaff['attractive'].value
        u = skaff['U'].value
        v = skaff['V'].value
        mutex_edges = np.stack([u, v], axis=1)
        mutex_edge_weights = skaff['W'].value
        shape = att_aff.shape[1:]

    offsets = [[-1, 0], [0, -1]]
    segmentation = nonlocal_mws_segmentation(att_aff, offsets,
                                             mutex_edges, mutex_edge_weights)

    print(segmentation)
    print(len(np.unique(segmentation)))
    # import pdb; pdb.set_trace()

    # from embeddingutils.affinities import normalized_cosine_similarity, logistic_similarity
    # import matplotlib.pyplot as plt
    # emb = torch.ones((2, 11, 11)).float()
    # emb[0] = -1
    # emb[0, :5, :3] = 1
    # emb[1, :5, :3] = -1
    # emb[1, 7:9, 4:] = -2
    # emb[0, 4:, 7:9] = 2
    # emb = emb + torch.randn(emb.shape) / 5
    # emb = emb[:, None]
    # print('embedding shape:', emb.shape)

    # for e in emb:
    #     plt.imshow(e[0])
    #     plt.show()

    # # for aff in embedding_to_affinities(emb, offsets='default-2D', affinity_measure=euclidean_similarity):
    # #    print(aff.shape)
    # #    print(torch.max(aff), torch.min(aff))

    # offsets = ((0, 1), (1, 0), (0, 1), (1, 0))
    # segs = (mws_segmentation(emb, offsets=offsets, affinity_measure=logistic_similarity,
    #                          ATT_C=2, repulsive_strides=(1, 1)))
    # for s in segs:
    #     plt.imshow(s)
    #     plt.show()

    # segs = hdbscan_segmentation(emb, metric='euclidean', min_cluster_size=8, n_img_dims=2, coord_scales=(.1, .1))
    # for s in segs:
    #     plt.imshow(s)
    #     plt.show()
