from torch.nn.functional import cosine_similarity
import inferno.utils.torch_utils as thu
import torch
import torch.nn.functional as F
import numpy as np
from embeddingutils.affinities import *
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
import inferno.utils.train_utils as tu
import collections
try:
    from speedrun.log_anywhere import log_image
except ImportError:
    log_image = None
from .lovasz_losses import StableBCELoss, lovasz_hinge


class ScalarLoggingMixin:
    def __init__(self):
        """
        Mixin for logging scalars via the inferno trainer.
        """
        super(ScalarLoggingMixin, self).__init__()
        self.validation_averages = dict()
        self.current_validation_iteration = None
        self.registered_states = set()

    def save_scalar(self, name, value, trainer=None, meta_tag=None):
        if trainer is None:
            assert hasattr(self, 'trainer')
            trainer = self.trainer
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone()
        # add prefix ('training' or 'validation'), meta_tag if specified, to name
        name = self.add_prefix(name, trainer, meta_tag=meta_tag)
        if name not in self.registered_states:
            self.observe_state(name, trainer)
        if trainer.model.training:  # training
            trainer.update_state(name, value)
        else:  # validation
            # check if it is a new validation run
            if self.current_validation_iteration != trainer._last_validated_at_iteration:
                self.current_validation_iteration = trainer._last_validated_at_iteration
                self.validation_averages = dict()

            # check if average meter for name has already been initialized this run
            if name not in self.validation_averages:
                self.validation_averages[name] = tu.AverageMeter()

            self.validation_averages[name].update(value)
            trainer.update_state(name, self.validation_averages[name].avg)

    def observe_state(self, name, trainer):
        assert hasattr(trainer, 'logger')
        logger = trainer.logger
        assert hasattr(logger, 'observe_state')
        time = 'training' if trainer.model.training else 'validation'
        logger.observe_state(name, time)

    def add_prefix(self, name, trainer, meta_tag=None):
        if meta_tag is not None or (hasattr(self, 'meta_tag') and self.meta_tag is not None):
            meta_tag = meta_tag if meta_tag is not None else self.meta_tag
            name = meta_tag + '/' + name
        if trainer.model.training:
            return 'training_' + name
        else:
            return 'validation_' + name


class WeightedLoss(ScalarLoggingMixin, torch.nn.Module):

    def __init__(self, loss_weights=None, trainer=None, loss_names=None, meta_tag=None):
        super(WeightedLoss, self).__init__()
        self.loss_weights = loss_weights
        self.meta_tag = type(self).__name__ if meta_tag is None else meta_tag
        if isinstance(loss_weights, collections.Sized) and not isinstance(loss_weights, str):
            self.n_losses = len(loss_weights)
            self.enable_logging = True
        if loss_names is None and loss_weights is not None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.logging_enabled = False
        self.trainer = trainer
        self.validation_averages = None  # Used to keep track of averages during validation

    def forward(self, preds, labels):
        losses = self.get_losses(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            if self.loss_weights is not None and not isinstance(self.loss_weights, str):
                weight = self.loss_weights[i]
            elif self.loss_weights == 'average':
                weight = 1 / len(losses)
            else:
                weight = 1
            loss = loss + weight * current

        self.save_losses(losses)
        total_loss = loss.mean()
        self.save_scalar('_total', total_loss.detach())
        return total_loss

    def save_losses(self, losses):
        if self.trainer is None:
            return
        losses = [loss.detach().mean() for loss in losses]
        for name, value in zip(self.loss_names, losses):
            self.save_scalar(name, value)

    def __getstate__(self):  # TODO make this nicer
        """Return state values to be pickled."""
        # mydict = dict(self.__dict__)
        # mydict['trainer'] = None
        return {}


class SumLoss(WeightedLoss):
    GRAD_PREFIX = {
        'norm': 'norm',
        'max': 'max',
        'mean': 'mean',
    }

    def __init__(self, losses, ignore_weight_zero=True, grad_stats=None, loss_weights=None, loss_names=None,
                 split_pred='auto', split_target='auto',
                 **super_kwargs):
        assert isinstance(losses, collections.Iterable)
        if ignore_weight_zero and isinstance(loss_weights, (list, tuple)):
            ind = np.array([i for i, w in enumerate(loss_weights) if w != 0])
            losses, loss_weights, loss_names = [[obj[i] for i in ind] if isinstance(obj, (list, tuple)) else obj
                                                for obj in (losses, loss_weights, loss_names)]
        super(SumLoss, self).__init__(loss_weights=loss_weights, loss_names=loss_names, **super_kwargs)
        assert isinstance(losses, collections.Sized)
        self.losses = losses
        self.grad_stats = grad_stats
        assert grad_stats is None or all(stat in self.GRAD_PREFIX for stat in grad_stats), \
            f'Supported stats: {list(self.GRAD_PREFIX.keys())}. Got {grad_stats}'
        if self.grad_stats is not None:
            assert self.trainer is not None
        self.split_pred = split_pred
        self.split_target = split_target
        self.hook_handle = None

    def save_grad_stats(self, stat_name, values):
        # divide by loss weights to get unweighted grad norms that are comparable for different weights
        values = [value / w for value, w in zip(values, self.loss_weights)]
        for name, value in zip(self.loss_names, values):
            self.save_scalar(name + '_' + self.GRAD_PREFIX[stat_name], value, meta_tag=self.meta_tag + '_gradient')

    def hook(self, grad):
        grad = grad.detach()
        if 'norm' in self.grad_stats:
            # calculate mini-batch average of L2 norms of gradients on model prediction
            grad = grad.view(grad.size(0), grad.size(1), -1)
            grad_norms = torch.norm(grad, dim=2).mean(dim=1)
            # divide by loss weights to get unweighted grad norms that are comparable for different weights
            self.save_grad_stats('norm', grad_norms)
        if 'max' in self.grad_stats:
            # calculate the maximum gradient applied on a single pixel.
            grad_max = grad.view(grad.size(0), -1).abs().max(dim=1)[0]
            self.save_grad_stats('max', grad_max)
        if 'mean' in self.grad_stats:
            # calculate the mean gradients.
            grad_mean = grad.view(grad.size(0), -1).mean(dim=1)
            self.save_grad_stats('mean', grad_mean)
        self.hook_handle.remove()

    def get_losses(self, preds, labels):

        split_target = isinstance(labels, (list, tuple)) if self.split_target is 'auto' else self.split_pred
        if split_target:
            # apply the list of losses to the list of predictions
            assert len(labels) == len(self.losses), f'{len(labels)} != {len(self.losses)}'
            label_per_loss = labels
        else:
            label_per_loss = [labels] * len(self.losses)

        split_pred = isinstance(preds, (list, tuple)) if self.split_pred is 'auto' else self.split_pred
        if split_pred:
            # apply the list of losses to the list of predictions
            assert len(preds) == len(self.losses)
            pred_per_loss = preds
        elif self.grad_stats is None or not self.trainer.model.training:
            pred_per_loss = [preds] * len(self.losses)
        else:
            pred_per_loss = preds[None].repeat(self.n_losses, *((1,) * len(preds.shape)))
            self.hook_handle = pred_per_loss.register_hook(self.hook)

        result = []
        for pred, labels, loss in zip(pred_per_loss, label_per_loss, self.losses):
            result.append(loss(pred, labels))
        return result


class LossSegmentwiseFreeTags(WeightedLoss):
    DISTANCE_MEASURES = {
        'l1_norm': l1_distance,
        'mean_l1_norm': mean_l1_distance,
        'l2_norm': euclidean_distance,
        'squared_l2_norm': squared_euclidean_distance,
        'cosine_distance': cosine_distance,
        'angular_distance': None,  # TODO
    }
    LOSS_FUNCS = {
        'mse': lambda tensor: tensor**2,
        'l1': lambda tensor: tensor.abs(),
        'huber': lambda tensor: F.smooth_l1_loss(tensor, tensor.new_zeros(1), reduce='none'),
    }
    PUSH_WEIGHTINGS = ['vanilla', 'per_pixel']
    PULL_WEIGHTINGS = ['vanilla', 'per_pixel']

    def __init__(self, loss_weights=(1, 0.1), ignore_label=None,
                 push_distance_measure='mean_l1_norm', push_loss_func='mse', push_margin=0.25, push_weighting='vanilla',
                 pull_distance_measure='mean_l1_norm', pull_loss_func='mse', pull_margin=0.00, pull_weighting='vanilla',
                 regularization_loss=None,
                 use_cosine_distance=False,
                 **super_kwargs):

        if regularization_loss is None:
            assert len(loss_weights) == 2, f'{loss_weights}'
            super_kwargs = dict(loss_weights=loss_weights, loss_names=['push-loss', 'pull-loss'], **super_kwargs)
        else:
            assert len(loss_weights) == 3, f'{loss_weights}'
            super_kwargs = dict(loss_weights=loss_weights,
                                loss_names=['push-loss', 'embedding-regularization', 'pull-loss'],
                                **super_kwargs)

        super(LossSegmentwiseFreeTags, self).__init__(**super_kwargs)

        self.regularization_loss = regularization_loss
        assert self.regularization_loss is None or callable(regularization_loss)

        self.ignore_label = ignore_label
        assert self.ignore_label in [None, 0], \
            'Ignore label other that 0 not implemented'

        # for a bit of backwards compatibility (yeah, a lot of things are not)
        if use_cosine_distance:
            print("'use_cosine_distance' is deprecated.")
            push_distance_measure = 'cosine_distance'
            pull_distance_measure = 'cosine_distance'

        # set attributes for push loss
        self.push_distance_measure = push_distance_measure if callable(push_distance_measure) else \
            self.DISTANCE_MEASURES[push_distance_measure]
        self.push_loss_func = push_loss_func if callable(push_loss_func) else \
            self.LOSS_FUNCS[push_loss_func]
        self.push_margin = push_margin
        self.push_weighting = push_weighting
        assert callable(self.push_weighting) or self.push_weighting in self.PUSH_WEIGHTINGS

        # set attributes for pull loss
        self.pull_distance_measure = pull_distance_measure if callable(pull_distance_measure) else \
            self.DISTANCE_MEASURES[pull_distance_measure]
        self.pull_loss_func = pull_loss_func if callable(pull_loss_func) else \
            self.LOSS_FUNCS[pull_loss_func]
        self.pull_margin = pull_margin
        self.pull_weighting = pull_weighting
        assert callable(self.pull_weighting) or self.pull_weighting in self.PULL_WEIGHTINGS


    def get_push_weights(self, segment_sizes, n_segments, n_pixels, n_active_pixels):
        if callable(self.push_weighting):
            return self.push_weighting(
                segment_sizes=segment_sizes,
                n_segments=n_segments,
                n_pixels=n_pixels,
                n_active_pixels=n_active_pixels
            )
        if self.push_weighting == 'vanilla':
            # behaviour as previous to refactor of this loss
            n_comparisons = 0.5 * (n_segments-1) * n_segments
            return 0 if n_comparisons == 0 else n_comparisons ** -0.5
        if self.push_weighting == 'per_pixel':
            # return segment_sizes ** 0.5
            n_comparisons = 0.5 * (n_segments-1) * n_segments
            return segment_sizes * max(1, n_active_pixels * n_comparisons) ** -0.5
        assert False, 'push weighting not understood'

    def get_pull_weights(self, segment_sizes, n_segments, n_pixels, n_active_pixels):
        if callable(self.pull_weighting):
            return self.pull_weighting(
                segment_sizes=segment_sizes,
                n_segments=n_segments,
                n_pixels=n_pixels,
                n_active_pixels=n_active_pixels
            )
        if self.pull_weighting == 'vanilla':
            # behaviour as previous to refactor of this loss
            return 1 / max(1, n_active_pixels)
        if self.pull_weighting == 'per_pixel':
            # constant gradient per pixel independent of segment size is equivalent to weighting the means with
            # segment size.
            return 1
        assert False, 'pull weighting not understood'

    def push_loss(self, centroids, weights=1):
        # return zero if there is only one centroid
        if len(centroids.shape) < 3 or centroids.shape[2] <= 1:
            print("skipping push because number of centroids is too low")
            return centroids.new_zeros(1)[0]
        # shape: n_stack * tag_dim * n_segments
        n_stack, tag_dim, n_segments = centroids.shape
        # calculate the distance of all cluster combinations
        distance_matrix = self.push_distance_measure(
            centroids[:, :, :, None].repeat(1, 1, 1, n_segments),
            centroids[:, :, None, :].repeat(1, 1, n_segments, 1),
            dim=1
        )
        # select vectorized upper triangle of distance matrix
        upper_tri_index = torch.arange(1, n_segments * n_segments + 1) \
            .view(n_segments, n_segments) \
            .triu(diagonal=1).nonzero().transpose(0, 1)
        cluster_distances = distance_matrix[:, upper_tri_index[0], upper_tri_index[1]]

        # determine weights for loss on individual distances
        if isinstance(weights, torch.Tensor) and tuple(weights.shape) == (n_segments,):
            weight_matrix = weights[None] * weights[:, None]
            distance_weights = weight_matrix[None, upper_tri_index[0], upper_tri_index[1]]
            distance_weights = distance_weights.expand_as(cluster_distances)
        elif weights is not None:
            distance_weights = weights
        else:
            distance_weights = 1

        return (distance_weights * self.push_loss_func(F.relu(self.push_margin - cluster_distances))).sum()

    def pull_loss(self, embedding, centroids, weights=1):
        if embedding.shape[0] == 0:
            print('skipping pull because everything is ignored')
            return embedding.new_zeros(1)[0]
        return (weights * self.pull_loss_func(
            F.relu(self.pull_distance_measure(embedding, centroids, dim=1) - self.pull_margin)
        )).sum()

    def get_losses(self, preds, labels):
        if torch.is_tensor(labels):
            gt_segs = labels
        else:
            gt_segs = labels[0]
        if len(preds.shape) == len(labels.shape):  # no intermediate predictions
            preds = preds[:, None]

        pushes, reg_losses, pulls = [], [], []
        for gt_seg, embeddings in zip(gt_segs, preds):  # iterate over minibatch
            n_segments = torch.max(gt_seg).int().item() + 1
            gt_seg = gt_seg.long()
            assert gt_seg.shape[0] == 1, 'segmentation should have one channel only'

            # get rid of extra spatial dimensions
            gt_seg = gt_seg[0].flatten()
            embeddings = embeddings.view(*embeddings.shape[:2], -1)

            # set up dictionary with information used to compute the segment and pixel weights for pull and push
            weighting_info = dict(n_pixels=gt_seg.nelement())

            # mask away ignore regions
            if self.ignore_label is not None:
                n_segments -= 1
                mask = gt_seg.ne(self.ignore_label)
                gt_seg = gt_seg[mask] - 1
                embeddings = embeddings[:, :, mask]
                weighting_info['n_active_pixels'] = mask.long().sum().item()
            weighting_info['n_segments'] = n_segments
            # calculate centroids and segment sizes of the individual segments
            centroids = []
            segment_sizes = []
            for seg_id in range(n_segments):
                segment_mask = gt_seg == seg_id
                segment_sizes.append(segment_mask.float().sum())
                centroids.append(embeddings[:, :, segment_mask].mean(-1))
            if not centroids:  # no segments present
                pushes.append(embeddings.new_zeros(1)[0])
                pulls.append(embeddings.new_zeros(1)[0])
                print('skipping: no segments present')
                continue
            centroids = torch.stack(centroids, dim=-1)
            segment_sizes = torch.stack(segment_sizes)
            weighting_info['segment_sizes'] = segment_sizes

            # push loss
            push_weights = self.get_push_weights(**weighting_info)
            push = self.push_loss(centroids, weights=push_weights)
            pushes.append(push)

            # regularization loss
            if self.regularization_loss is not None:
                reg_loss = self.regularization_loss(centroids)
                reg_losses.append(reg_loss)

            # pull loss
            pixel_wise_centroids = centroids[:, :, gt_seg]
            pull_weights = self.get_pull_weights(**weighting_info)
            pull = self.pull_loss(embeddings, pixel_wise_centroids, weights=pull_weights)
            pulls.append(pull)

        if reg_losses:
            return torch.stack(pushes), torch.stack(pulls), torch.stack(reg_losses)
        else:
            return torch.stack(pushes), torch.stack(pulls)


class LossAffinitiesFromEmbedding(WeightedLoss):
    def __init__(self, offsets='default-3D', loss_weights=None, ignore_label=None, margin=0,
                 use_cosine_distance=False, pull_weight=0, push_weight=0, affinity_weight=1,
                 affinities_direct=False, **super_kwargs):
        if callable(offsets):
            self.offset_sampler = offsets
            self.dynamic_offsets = True
            offsets = self.offset_sampler()
            loss_names = None
            loss_weights = 'average'
        else:
            self.offsets = get_offsets(offsets)
            self.dynamic_offsets = False
            if loss_weights is None:
                loss_weights = (1 / len(self.offsets),) * len(self.offsets)
            assert len(loss_weights) == len(self.offsets)
            loss_names = ["offset_" + '_'.join(str(o) for o in off) for off in self.offsets]
            print(loss_names)

        super(LossAffinitiesFromEmbedding, self).__init__(
            loss_weights=loss_weights,
            loss_names=loss_names,
            enable_logging=not self.dynamic_offsets,
            **super_kwargs)
        self.ignore_label = ignore_label
        self.use_cosine_distance = use_cosine_distance
        self.ignore_label = ignore_label
        self.margin = margin
        self.push_weight = push_weight
        self.pull_weight = pull_weight
        self.affinity_weight = affinity_weight

        # initialize distance/affinity generating functions
        self.seg_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                affinity_measure=label_equal_similarity,
                                                pass_offset=False)

        if self.affinity_weight is not 0:
            self.emb_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                    affinity_measure=self.affinity_measure,
                                                    pass_offset=True)
            self.aff_loss = SorensenDiceLoss(channelwise=False)

        if self.ignore_label is not None:
            self.seg_to_mask = EmbeddingToAffinities(offsets=offsets,
                                                     affinity_measure=ignore_label_mask_similarity,
                                                     pass_offset=False)

        if self.push_weight != 0 or self.pull_weight != 0:
            self.emb_to_dist = EmbeddingToAffinities(offsets=offsets,
                                                     affinity_measure=self.distance_measure,
                                                     pass_offset=False)

        self.affinities_direct = affinities_direct
        if affinities_direct:
            assert self.pull_weight == self.push_weight == 0 and self.affinity_weight != 0
        self.relu = torch.nn.ReLU()

    def set_offsets(self, offsets):
        self.offsets = offsets
        if self.affinity_weight != 0:
            self.seg_to_aff.offsets = offsets
            self.emb_to_aff.offsets = offsets
        if self.pull_weight != 0 or self.push_weight != 0:
            self.emb_to_dist.offsets = offsets
        if self.ignore_label is not None:
            self.seg_to_mask.offsets = offsets

    def push_loss(self, dists):
        # return (self.relu(self.margin - dists)).mean()
        return (self.relu(self.margin - dists) ** 2).mean()

    def pull_loss(self, dists):
        # return (dists).mean()
        return (dists ** 2).mean()

    def affinity_measure(self, x, y, dim, offset):
        if self.use_cosine_distance:
            return self.relu(normalized_cosine_similarity(x, y, dim=dim) * 2 - 1)
        else:
            return logistic_similarity(x, y, dim=dim)  # =offset/100)#

    def distance_measure(self, x, y, dim):
        if self.use_cosine_distance:
            return normalized_cosine_similarity(x, y)
        else:
            return euclidean_distance(x, y, dim)

    def get_losses(self, preds, labels):
        # check if random offsets are used and if yes, sample them
        if self.dynamic_offsets:
            offsets = self.offset_sampler()
            self.set_offsets(offsets)
        else:
            offsets = self.offsets

        if torch.is_tensor(labels):
            gt_segs = labels
        else:
            gt_segs = labels[0]

        if len(preds.shape) == len(labels.shape):  # no intermediate predictions
            preds = preds[:, None]

        with torch.no_grad():
            gt_aff = self.seg_to_aff(gt_segs)

        if self.affinity_weight != 0:
            if not self.affinities_direct:
                pred_aff = self.emb_to_aff(preds)
            else:
                pred_aff = preds

        if self.push_weight != 0 or self.pull_weight != 0:
            pred_dist = self.emb_to_dist(preds)

        if self.ignore_label is not None:
            masks = self.seg_to_mask(gt_segs)
        else:
            masks = torch.ones_like(gt_segs).byte()

        losses_per_offset = []
        for j, offset in enumerate(offsets):  # iterate over offsets
            loss_this_offset = []
            for i in range(preds.shape[1]):  # iterate over intermediate outputs
                current_loss = torch.tensor(0).float().to(preds.device)
                if self.affinity_weight != 0:
                    ind = masks[:, j]
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += -(1 - m) + m * self.affinity_weight * \
                                        self.aff_loss(1 - pred_aff[:, i, j][ind], 1 - gt_aff[:, j][ind])
                if self.push_weight != 0:
                    ind = masks[:, j] * (gt_aff[:, j]).byte() != 0
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += m * self.push_weight * \
                                        self.push_loss(pred_dist[:, i, j][ind])
                if self.pull_weight != 0:
                    ind = masks[:, j] * (1 - gt_aff[:, j]).byte() != 0
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += m * self.pull_weight * \
                                        self.pull_loss(pred_dist[:, i, j][ind])
                loss_this_offset.append(current_loss)
            losses_per_offset.append(torch.stack(loss_this_offset, dim=0))

        return losses_per_offset


class L2RegularizationLoss(torch.nn.Module):
    def forward(self, pred, *labels):
        return (pred ** 2).sum()


class NevenMaskLoss(WeightedLoss):
    """
    Generalization of loss from
    "Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth" by Neven et al.

    Notes on the original code at https://github.com/davyneven/SpatialEmbeddings
        - Model is a multi branched variant with 3 independent decoders for embeddings, sigmas, seeds.
        - Last layer of model is initialized differently: sigmas to 0 weight, 1 bias, seeds to 0.
        - Sigma is made positive by sigma = exp(10 * sigma). Why 10?
        - Clustering:
            - They only cluster all but 128 (seems arbitrary) pixels are assigned.

    Ideas for generalization:
        - Free, non-offset embeddings
        - learned sigma -> learned, position dependent similarity measure
            - general gaussians
            - weights of a couple of 1x1 convolutions
    """
    BINARY_LOSSES = {
        'BCE': torch.nn.BCELoss(),
        'Dice': SorensenDiceLoss(channelwise=True),
        'LovaszHinge': lambda pred, label: lovasz_hinge(2 * pred - 1, label, per_image=True),
    }
    def __init__(self, loss_weights=(1, 1, 1), ignore_label=None, log_masks=0, binary_loss='BCE', spatial_dim=3,
                 **super_kwargs):

        super_kwargs = dict(loss_weights=loss_weights,
                            loss_names=['instance-loss', 'variance-loss', 'seed-loss'],
                            **super_kwargs)

        super(NevenMaskLoss, self).__init__(**super_kwargs)
        assert binary_loss in self.BINARY_LOSSES, f'Please select one of {self.BINARY_LOSSES}. Got {binary_loss}.'
        self.mask_comparison_loss = self.BINARY_LOSSES[binary_loss]

        self.ignore_label = ignore_label
        self.log_masks = log_masks
        self.predicted_masks = None
        self.gt_masks = None
        self.spatial_dim = spatial_dim

    def get_losses(self, preds, labels):
        embeddings, sigmas, seed_maps = preds
        gt_segs = labels[0]

        # for logging purposes only
        self.predicted_masks = []
        self.gt_masks = []

        losses = [self.get_losses_single_sample(*inputs) for inputs in zip(embeddings, sigmas, seed_maps, gt_segs)]
        if self.log_masks:
            assert log_image is not None, f'need speedrun anywhere logging to log masks'
            log_image('pred_instance_masks', torch.stack(self.predicted_masks, dim=1))
            log_image('gt_instance_masks', torch.stack(self.gt_masks, dim=1))
        return torch.stack(losses).mean(0)

    def get_losses_single_sample(self, embedding, sigmas, seed_map, gt_seg):
        instance_ids = gt_seg.unique()
        instance_ids = instance_ids[instance_ids != self.ignore_label]
        losses = [self.get_loss_single_instance(instance_id, embedding, sigmas, seed_map, gt_seg)
                  for instance_id in instance_ids]
        losses = torch.stack([torch.stack(loss) for loss in losses])

        # regress seeds to 0 on background
        if self.ignore_label is not None:
            losses[-1] += ((seed_map[gt_seg == self.ignore_label]) ** 2).mean()
        return losses.mean(0)

    def predicted_instance_mask(self, embedding, target_embedding, sigma):
        return torch.exp(-((embedding - target_embedding[(slice(None),) + self.spatial_dim*(None,)]) ** 2).sum(0, keepdim=True) / sigma ** 2)

    def get_loss_single_instance(self, instance_id, embedding, sigmas, seed_maps, gt_seg):
        instance_mask = (gt_seg == instance_id)[0]  # (1, W, H)

        # the target embedding is the mean of embedding vectors of pixels in instance
        target_embedding = embedding[:, instance_mask].view(embedding.shape[0], -1).mean(1)  # (E)

        # we take as sigma the mean predicted sigma over the instance
        sigma = sigmas[:, instance_mask].view(sigmas.shape[0], -1).mean(1)

        predicted_mask = self.predicted_instance_mask(embedding, target_embedding, sigma)
        if len(self.predicted_masks) < self.log_masks:
            self.predicted_masks.append(predicted_mask.detach()[None])  # B I D H W
            self.gt_masks.append(instance_mask.detach()[None, None])    # B I D H W

        instance_loss = self.mask_comparison_loss(predicted_mask, instance_mask.float()[None])

        sigma_loss = ((sigmas[:, instance_mask] - sigma.detach().item()) ** 2).sum(0).mean()

        seed_maps_loss = ((seed_maps[:, instance_mask] - predicted_mask[:, instance_mask]) ** 2).sum(0).mean()

        return instance_loss, sigma_loss, seed_maps_loss


class NevenMaskLossParallel(NevenMaskLoss):

    def predicted_instance_mask(self, embedding, target_embedding, sigma):
        return torch.exp(-((embedding - target_embedding[(slice(None),) + self.spatial_dim*(None,)]) ** 2)
                         .sum(0, keepdim=True) / sigma ** 2)

    def get_losses_single_sample(self, embedding, sigmas, seed_map, gt_seg):
        instance_ids = gt_seg.unique()
        instance_ids = instance_ids[instance_ids != self.ignore_label]

        # compute ground truth instance masks
        instance_masks = gt_seg.eq(instance_ids[(slice(None),) + (None,) * self.spatial_dim])  # I D H W

        # compute target embeddings as average over segment
        target_embeddings = [embedding[:, instance_mask].view(embedding.shape[0], -1).mean(1)
                             for instance_mask in instance_masks]
        target_embeddings = torch.stack(target_embeddings)  # I, E

        # compute predicted sigma values
        mean_sigmas = [sigmas[:, instance_mask].view(sigmas.shape[0], -1).mean(1)
                       for instance_mask in instance_masks]
        mean_sigmas = torch.stack(mean_sigmas)  # I, n_sigma

        embedding_dists = embedding - target_embeddings[(slice(None),) * 2 + (None,) * self.spatial_dim]  # I E D H W
        predicted_masks = torch.exp(-(embedding_dists ** 2).sum(1, keepdim=True)
                                    / mean_sigmas.view(-1, 1, *((1,)*self.spatial_dim)) ** 2)  # I 1 D H W

        instance_loss = self.mask_comparison_loss(predicted_masks[:, 0].clamp(min=0, max=1), instance_masks.float())

        sigma_loss = [((sigmas[:, instance_mask] - mean_sigma.detach().item()) ** 2).sum(0).mean()
                      for instance_mask, mean_sigma in zip(instance_masks, mean_sigmas)]
        sigma_loss = torch.stack(sigma_loss).mean()

        target_seed_map = (predicted_masks * instance_masks.float()[:, None]).sum(0)
        seed_maps_loss = ((seed_map - target_seed_map) ** 2).mean()

        if self.log_masks and not self.predicted_masks:
            self.predicted_masks.extend(predicted_masks.detach()[:self.log_masks, None, 0])  # B I D H W
            self.gt_masks.extend(instance_masks.detach()[:self.log_masks, None])  # B I D H W

        return torch.stack([instance_loss, sigma_loss, seed_maps_loss])

    def get_loss_single_instance(self, instance_id, embedding, sigmas, seed_maps, gt_seg):
        raise NotImplementedError
