import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class StardistLoss(nn.Module):
    """Loss for stardist predictions combines BCE loss for probabilities
       with MAE (L1) loss for distances

    Args:
        weight: Distance loss weight. Total loss will be bce_loss + weight * l1_loss
    """

    def __init__(self, weight=1.):

        super().__init__()
        self.weight = weight

    def forward(self, prediction, target, mask=None):
        # Predicted distances errors are weighted by object prob
        if target.shape!=prediction.shape:
            prediction = prediction.squeeze(1)

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
                                                         target_prob[ignore_mask].float(),
                                                         reduction='sum') / imsum
            return self.weight * l1loss + bceloss

        # weight predictions by target probs
        l1loss = (target_prob * l1loss_pp).mean()

        bceloss = F.binary_cross_entropy_with_logits(predicted_prob,
                                                     target_prob.float(),
                                                     reduction='mean')

        return (self.weight * l1loss) + bceloss