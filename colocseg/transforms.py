import numpy as np
import stardist
from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
from inferno.io.transform import Transform
from cellpose.dynamics import masks_to_flows

class ThreeclassTf(Transform):
    """Convert segmentation to 3 class"""

    def __init__(self, inner_distance=4):
        super().__init__()
        self.inner_distance = inner_distance

    def tensor_function(self, gt):
        gt_stardist = stardist.geometry.star_dist(gt, n_rays=8)
        background = gt == 0
        inner = gt_stardist.min(axis=-1) > self.inner_distance
        # classes 0: boundaries, 1: inner_cell, 2: background
        threeclass = (2 * background) + inner
        return threeclass.astype(np.long)


class StardistTf(Transform):
    """Convert segmentation to stardist"""

    def __init__(self, n_rays=16, fill_label_holes=False):
        super().__init__()
        self.n_rays = n_rays
        self.fill_label_holes = fill_label_holes

    def tensor_function(self, gt):
        # gt = measure.label(gt)
        if self.fill_label_holes:
            gt = stardist.fill_label_holes(gt)
        dist = stardist.geometry.star_dist(gt, n_rays=self.n_rays)
        dist_mask = stardist.utils.edt_prob(gt)
        
        if gt.min() < 0:
            # ignore label found
            ignore_mask = gt < 0
            dist[ignore_mask] = 0
            dist_mask[ignore_mask] = -1

        dist_mask = dist_mask[None]
        dist = np.transpose(dist, (2, 0, 1))

        mask_and_dist = np.concatenate([dist_mask, dist], axis=0)
        return mask_and_dist


def get_offsets():
    return np.array(((-1, 0),
                     (0, -1),
                     (0., -4.),
                     (-3., -3.),
                     (-4., -0.),
                     (-3., 3.),
                     (0, -8),
                     (-8, 0)), int)


class AffinityTf(Transform):
    """Convert segmentation to stardist"""

    def __init__(self, offsets=None):
        super().__init__()
        self.offsets = get_offsets() if offsets is None else offsets

        self.seg2aff = Segmentation2AffinitiesWithPadding(
            self.offsets,
            retain_segmentation=False,
            segmentation_to_binary=False,
            ignore_label=-1)

    def tensor_function(self, gt):
        aff = 1 - self.seg2aff.tensor_function(gt).astype(np.float32)
        return np.concatenate(((gt > 0)[None].astype(aff.dtype), aff), axis=0)


class CellposeTf(Transform):
    """Convert segmentation to cellpose targets
       target: ND - array, float
       transformed labels in array[nimg x nchan x xy[0] x xy[1]]
       target[:, 0] distance fields
       target[:, 1:3] flow fields
       target[:, 3] boundary fields
       target[:, 4] boundary - emphasized weights"""

    def __init__(self):
        super().__init__()

    def tensor_function(self, gt):

        ignore_mask = gt < 0
        gt = gt.copy()
        gt[ignore_mask] = 0

        seg, distance_field, probs, flow = masks_to_flows(gt)
        # boundary map
        bd = (distance_field==1)
        bd[bd==0] = 0
        # cell mask
        w = 0.1 + (seg>0)
        w[ignore_mask] = 0

        out = np.concatenate([distance_field[None],
                              flow,
                              bd[None],
                              w[None]], axis=0)
        out = out.astype(np.float32)
        return out

        
        # trgt = dynamics.masks_to_flows(train_labels[0], dists=None, use_gpu=False, device=None, omni=True)
        # dist = trgt[1]
        # heat = trgt[2]
        # flow_fields = trgt[3]
        
        # label, label > 0, utils.distance_to_boundary(label)),

        # mask = lbl[6] = train_labels[0] > 0
        # bg_edt = edt.edt(mask<0.5,black_border=True) #last arg gives weight to the border, which seems to always lose
        # cutoff = 9
        # lbl[7] = (gaussian(1-np.clip(bg_edt,0,cutoff)/cutoff, 1)+0.5)

        # return mask_and_dist
