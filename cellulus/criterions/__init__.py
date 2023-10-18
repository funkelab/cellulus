from cellulus.criterions.oce_loss import OCELoss
import numpy as np
import stardist
from inferno.io.transform import Transform
import gunpowder as gp


def get_loss(
    temperature, regularizer_weight, density, kappa, num_spatial_dims, reduce_mean
):
    return OCELoss(
        temperature, regularizer_weight, density, kappa, num_spatial_dims, reduce_mean
    )

class TransformStardist(gp.BatchFilter):
            def __init__(self,array):
                self.array = array
            def prepare(self, request):

                # the requested ROI for array
                # expects (17,x,y)
                roi = request[self.array].roi

                self.stardist_shape = roi.get_shape()
                self.stardist_roi = roi
                print('roi = ',roi)

                # 1. compute the context
                # context = gp.Coordinate((self.truncate,)*roi.dims()) * self.sigma

                # 2. enlarge the requested ROI by the context
                # roi.__offset = [0,0,0,0]
                # context_roi = roi.set_shape([1,1,stardist_shape[1],stardist_shape[2]])
                # context_roi = gp.Roi((0,0,0,0),(1,1,self.stardist_shape[1],self.stardist_shape[2]))
                roi = gp.Roi((0,0,0,0),(1,1,self.stardist_shape[1],self.stardist_shape[2]))
                print('roi =',roi)

                # create a new request with our dependencies
                deps = gp.BatchRequest()
                deps[self.array] = roi
                print('deps created')
                # return the request
                return deps
            
            def process(self, batch, request):
                self.data_shape = data.shape
                data = batch[self.array].data
                # import numpy as np
                print(self.array, data.shape, np.unique(data))
                temp = stardist_transform(data)
                print(temp.shape, np.unique(temp))
                batch[self.array].data = temp


def stardist_transform(gt, n_rays=16, fill_label_holes=False):

    if len(gt.shape)>2:
         gt = np.squeeze(gt)

    if np.any(gt - gt.astype(np.uint16)):
            mapping={v:k for k,v in enumerate(np.unique(gt))}
            u,inv = np.unique(gt,return_inverse = True)
            Y1 = np.array([mapping[x] for x in u])[inv].reshape(gt.shape)
            gt = Y1.astype(np.uint16)


    if fill_label_holes:
        gt = stardist.fill_label_holes(gt)

    dist = stardist.geometry.star_dist(gt, n_rays = n_rays)
    dist_mask = stardist.utils.edt_prob(gt.astype(int))
    
    if gt.min() < 0:
        # ignore label found
        ignore_mask = gt < 0
        print(gt.shape, dist.shape)
        dist[ignore_mask] = 0
        dist_mask[ignore_mask] = -1

    dist_mask = dist_mask[None]
    dist = np.transpose(dist, (2, 0, 1))

    # dist_mask = torch.tensor(dist_mask)
    # dist = torch.tensor(dist)
    mask_and_dist = np.concatenate([dist_mask, dist], axis=0)

    # mask_and_dist = torch.cat([dist_mask, dist], axis=0)
    return mask_and_dist


class StardistTf(Transform):
    """Convert segmentation to stardist"""

    def __init__(self, n_rays=16, fill_label_holes=False):
        super().__init__()
        self.n_rays = n_rays
        self.fill_label_holes = fill_label_holes

    def tensor_function(self, gt):

        if np.any(gt-gt.astype(np.uint16)):
            mapping={v:k for k,v in enumerate(np.unique(gt))}
            u,inv = np.unique(gt,return_inverse = True)
            Y1 = np.array([mapping[x] for x in u])[inv].reshape(gt.shape)
            gt = Y1.astype(np.uint16)
        # gt = measure.label(gt)
        if self.fill_label_holes:
            gt = stardist.fill_label_holes(gt)
        # import pdb
        # pdb.set_trace()
        # print('gt.type',gt.type())
        dist = stardist.geometry.star_dist(gt, n_rays=self.n_rays)
        dist_mask = stardist.utils.edt_prob(gt)
        
        if gt.min() < 0:
            # ignore label found
            ignore_mask = gt < 0
            print(gt.shape, dist.shape)
            dist[ignore_mask] = 0
            dist_mask[ignore_mask] = -1

        dist_mask = dist_mask[None]
        dist = np.transpose(dist, (2, 0, 1))

        mask_and_dist = np.concatenate([dist_mask, dist], axis=0)
        return mask_and_dist