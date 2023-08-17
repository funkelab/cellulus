import gc
import time
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch
from inferno.extensions.criteria.set_similarity_measures import \
    SorensenDiceLoss
from matplotlib import collections as mc
from skimage.io import imsave
from torch import Tensor
from torch.nn import functional as F
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.nn.modules.module import Module
from cellpose.core import MaskedLoss, DerivativeLoss, WeightedLoss, ArcCosDotLoss, NormLoss, DivergenceLoss
from colocseg.utils import cluster_embeddings, label2color

matplotlib.use('agg')


class AnchorLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is
            smaller than the anchor_radius
    """

    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature

    def distance_fn(self, e0, e1):
        diff = (e0 - e1)
        return diff.norm(2, dim=-1)

    def nonlinearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, anchor_embedding, reference_embedding) -> Tensor:
        # compute all pairwise distances of anchor embeddings and reference embedding
        # reference embedding are detached, to avoid biased gradients at the image boundaries
        dist = self.distance_fn(anchor_embedding, reference_embedding.detach())
        # dist.shape = (b, p, p)
        nonlinear_dist = self.nonlinearity(dist)
        return nonlinear_dist.sum()

    def absoute_embedding(self, embedding, abs_coords):
        return embedding + abs_coords


class AnchorPlusContrastiveLoss(AnchorLoss):

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.weight = 10.

    def forward(self, embedding, contr_emb, abs_coords, patch_mask) -> Tensor:
        # compute all pairwise distances of anchor embeddings
        dist = self.distance_fn(embedding, abs_coords)
        # dist.shape = (b, p, p)

        nonlinear_dist = self.nonlinearity(dist)
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        nonlinear_dist = nonlinear_dist[patch_mask == 1]

        loss = nonlinear_dist.sum()

        if contr_emb is None:
            return loss

        try:
            cluster_labels = cluster_embeddings(embedding + abs_coords)
            contr_emb = F.normalize(contr_emb, dim=-1)
            cum_mean_clusters = []

            for b in range(len(embedding)):
                with torch.no_grad():
                    mean_clusters = [contr_emb[b, cluster_labels[b] == i].mean(
                        axis=0) for i in np.unique(cluster_labels[b]) if i >= 0]
                    if len(mean_clusters) > 0:
                        mean_clusters = torch.stack(mean_clusters, dim=-1)
                        cum_mean_clusters.append(mean_clusters)

            cum_mean_clusters = torch.cat(cum_mean_clusters, dim=-1)
            stacked_contr_emb = contr_emb.view(-1, cum_mean_clusters.shape[0])
            logits = torch.matmul(stacked_contr_emb, cum_mean_clusters)
            target = torch.from_numpy(np.concatenate(cluster_labels, axis=0)).long().to(logits.device)
            bce_loss = self.ce(logits, target)
            loss += self.weight * bce_loss
        except:
            print("clustering failed! Returning anchor loss")

        return loss


class StardistLoss(torch.nn.Module):
    """Loss for stardist predicsions combines BCE loss for probabilities
       with MAE (L1) loss for distances

    Args:
        weight: Distance loss weight. Total loss will be bce_loss + weight * l1_loss
    """

    def __init__(self, weight=1.):

        super().__init__()
        self.weight = weight

    def forward(self, prediction, target, mask=None):
        # Predicted distances errors are weighted by object prob
        target_prob = target[:, :1]
        predicted_prob = prediction[:, :1]
        target_dist = target[:, 1:]
        predicted_dist = prediction[:, 1:]

        if mask is not None:
            target_prob = mask * target_prob
            # do not train foreground prediction when mask is supplied
            predicted_prob = predicted_prob.detach()

        l1loss_pp = F.l1_loss(predicted_dist,
                           target_dist,
                           reduction='none')
        
        ignore_mask_provided = target_prob.min() < 0
        if ignore_mask_provided:
            # ignore label was supplied
            ignore_mask = target_prob >= 0.
            # add one to avoid division by zero
            imsum = ignore_mask.sum()
            if imsum == 0:
                print("WARNING: Batch with only ignorelabel encountered!")
                return 0*l1loss_pp.sum()

            l1loss = ((target_prob * ignore_mask) * l1loss_pp).sum() / imsum
            bceloss = F.binary_cross_entropy_with_logits(predicted_prob[ignore_mask],
                                                         target_prob[ignore_mask],
                                                         reduction='sum') / imsum
            return self.weight * l1loss + bceloss

        # weight predictions by target probs
        l1loss = (target_prob * l1loss_pp).mean()
        bceloss = F.binary_cross_entropy_with_logits(predicted_prob,
                                                     target_prob,
                                                     reduction='mean')
        return self.weight * l1loss + bceloss


class RegressionLoss(torch.nn.Module):
    """MAE (L1) regression loss"""

    def forward(self, prediction, target, mask=None):

        if mask is not None:
            target = mask * target
            prediction = mask * prediction

        l1loss = F.l1_loss(prediction,
                           target)
        return l1loss


class AffinityLoss(torch.nn.Module):
    """Loss for affiniy predicsions combines SorensenDiceLoss loss for affinities
       with BCE loss for foreground background

        Args:
            weight: affinity loss weight. Total loss will be bce_loss + weight * aff_loss
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.sd_loss = SorensenDiceLoss()
        self.fgbg_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        bceloss = self.fgbg_loss(prediction[:, :1], target[:, :1])
        aff_loss = self.sd_loss(prediction[:, 1:].sigmoid(), target[:, 1:])
        return self.weight * aff_loss + bceloss


class CellposeLoss(torch.nn.Module):
    """Loss for cellpose flow predictions
       adapted from https://github.com/MouseLand/cellpose
    """

    def __init__(self):
        super().__init__()
        self.criterion  = MSELoss(reduction='mean')
        self.criterion2 = BCEWithLogitsLoss(reduction='mean')
        self.criterion6 = MaskedLoss()
        self.criterion11 = DerivativeLoss()
        self.criterion12 = WeightedLoss()
        self.criterion14 = ArcCosDotLoss()
        self.criterion15 = NormLoss()
        self.criterion16 = DivergenceLoss()

    def forward(self, prediction, target):
        """ Loss function for Omnipose.
            
            Parameters
            --------------
            target: ND-array, float
                transformed labels in array [nimg x nchan x xy[0] x xy[1]]
                target[:,0] distance fields
                target[:,1:3] flow fields 
                target[:,3] boundary fields
                target[:,4] boundary-emphasized weights        
            
            prediction:  ND-tensor, float
                network predictions
                prediction[:,:2] flow fields
                prediction[:,2] distance fields
                prediction[:,3] boundary fields
            
            """
            
        veci = target[:,1:3]
        dist = target[:,0]
        boundary =  target[:,3]
        cellmask = dist>0
        w =  target[:,4]
        dist = dist
        boundary = boundary
        cellmask = cellmask.bool()

        flow = prediction[:,:2] # 0,1
        dt = prediction[:,2]
        bd = prediction[:,3]
        a = 10.

        wt = torch.stack((w,w),dim=1)
        ct = torch.stack((cellmask,cellmask),dim=1) 

        loss1 = 10.*self.criterion12(flow,veci,wt)  #weighted MSE 
        loss2 = self.criterion14(flow,veci,w,cellmask) #ArcCosDotLoss
        loss3 = self.criterion11(flow,veci,wt,ct)/a # DerivativeLoss
        loss4 = 2.*self.criterion2(bd,boundary)
        loss5 = 2.*self.criterion15(flow,veci,w,cellmask) # loss on norm 
        loss6 = 2.*self.criterion12(dt,dist,w) #weighted MSE 
        loss7 = self.criterion11(dt.unsqueeze(1),
                                 dist.unsqueeze(1),
                                 w.unsqueeze(1),
                                 cellmask.unsqueeze(1))/a  

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
