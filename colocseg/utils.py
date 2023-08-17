import inspect
import json
import numbers
import os
from time import time
import random
from argparse import ArgumentParser
from functools import partial

import gunpowder as gp
import imgaug as ia
import matplotlib
import numpy as np
import scipy
import scipy.sparse as sparse
import torch
from gunpowder.batch_request import BatchRequest
from imgaug import augmenters as iaa
from inferno.io.transform.base import Transform
from PIL import Image
from pytorch_lightning.callbacks import Callback
from scipy import ndimage
from skimage import measure
from skimage.filters import rank
from skimage.measure import label, regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.transform import rescale
from sklearn.cluster import DBSCAN
from torch.nn import functional as F
from torch.utils.data import Sampler


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


def label2color(label):

    if isinstance(label, Image.Image):
        label = np.array(label)
        if len(label.shape) == 3:
            label = label[..., 0]

    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    shuffle_labels = np.concatenate(
        ([0], np.random.permutation(label.max()) + 1))
    label = shuffle_labels[label]
    return cmap(label / (label.max() + 1)).transpose(2, 0, 1)


def try_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def visnorm(x):
    x = x - x.min()
    x = x / x.max()
    return x


def vis(x, normalize=True):
    if isinstance(x, Image.Image):
        x = np.array(x)

    assert(len(x.shape) in [2, 3])

    if len(x.shape) == 2:
        x = x[None]
    else:
        if x.shape[0] not in [1, 3]:
            if x.shape[2] in [1, 3]:
                x = x.transpose(2, 0, 1)
            else:
                raise Exception(
                    "can not visualize array with shape ", x.shape)

    if normalize:
        with torch.no_grad():
            visnorm(x)

    return x


def log_img(name, img, pl_module):
    pl_module.logger.experiment.add_image(name, img, pl_module.global_step)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except BaseException:
        return False


def save_args(args, directory):
    os.mkdir(directory)
    log_out = os.path.join(directory, "commandline_args.txt")
    serializable_args = {key: value for (key, value) in args.__dict__.items() if is_jsonable(value)}

    with open(log_out, 'w') as f:
        json.dump(serializable_args, f, indent=2)


def adapted_rand(seg, gt, all_stats=False, ignore_label=True):
    """Compute Adapted Rand error.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    ignore_label: boolean, optional
        whether to ignore the zero label
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = int(np.amax(segA)) + 1
    n_labels_B = int(np.amax(segB)) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix(
        (ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    if ignore_label:
        a = p_ij[1:n_labels_A, :]
        b = p_ij[1:n_labels_A, 1:n_labels_B]
        c = p_ij[1:n_labels_A, 0].todense()
    else:
        a = p_ij[:n_labels_A, :]
        b = p_ij[:n_labels_A, 1:n_labels_B]
        c = p_ij[:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return {"are": are,
                "precision": precision,
                "recall": recall}
    else:
        return are


def offset_from_direction(direction, max_direction=8., distance=10):
    angle = (direction / max_direction)
    angle = 2 * np.pi * angle

    x_offset = int(0.75 * distance * np.sin(angle))
    y_offset = int(0.75 * distance * np.cos(angle))

    x_offset += random.randint(-int(0.15 * distance),
                               +int(0.15 * distance))
    y_offset += random.randint(-int(0.15 * distance),
                               +int(0.15 * distance))

    return x_offset, y_offset


def random_offset(distance=10):
    angle = 2 * np.pi * np.random.uniform()
    distance = np.random.uniform(low=1., high=distance)

    x_offset = int(distance * np.sin(angle))
    y_offset = int(distance * np.cos(angle))

    return x_offset, y_offset

# if y_hat.requires_grad:
#     def log_hook(grad_input):
#         # torch.cat((grad_input.detach().cpu(), y_hat.detach().cpu()), dim=0)
#         grad_input_batch = torch.cat(tuple(torch.cat(tuple(vis(e_0[c]) for c in range(e_0.shape[0])), dim=1) for e_0 in grad_input), dim=2)
#         self.logger.experiment.add_image(f'train_regression_grad', grad_input_batch, self.global_step)
#         handle.remove()

#     handle = y_hat.register_hook(log_hook)


class UpSample(gp.nodes.BatchFilter):

    def __init__(self, source, factor, target):

        assert isinstance(source, gp.ArrayKey)
        assert isinstance(target, gp.ArrayKey)
        assert (
            isinstance(factor, numbers.Number) or isinstance(factor, tuple)), (
            "Scaling factor should be a number or a tuple of numbers.")

        self.source = source
        self.factor = factor
        self.target = target

    def setup(self):

        spec = self.spec[self.source].copy()
        spec.roi = spec.roi * self.factor
        self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):

        deps = gp.BatchRequest()
        sdep = request[self.target]
        sdep.roi = sdep.roi / self.factor
        deps[self.source] = sdep
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        # logger.debug("upsampeling %s with %s", self.source, self.factor)

        # resize
        data = batch.arrays[self.source].data
        data = rescale(data, self.factor)

        # create output array
        spec = self.spec[self.target].copy()
        spec.roi = request[self.target].roi
        outputs.arrays[self.target] = gp.Array(data, spec)

        return outputs


class AbsolutIntensityAugment(gp.nodes.BatchFilter):

    def __init__(self, array, scale_min, scale_max, shift_min, shift_max):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        raw.data = self.__augment(raw.data,
                                  np.random.uniform(low=self.scale_min, high=self.scale_max),
                                  np.random.uniform(low=self.shift_min, high=self.shift_max))

        # clip values, we might have pushed them out of [0,1]
        raw.data[raw.data > 1] = 1
        raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):

        return a * scale + shift


class Patchify(object):
    """ Adapted from
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/8a4cf8f61644c28d6df54ccffe3a52d6f5fce5a6/pl_bolts/transforms/self_supervised/ssl_transforms.py#L62
    This implementation adds a dilation parameter
    """

    def __init__(self, patch_size, overlap_size, dilation):
        self.patch_size = patch_size
        self.overlap_size = self.patch_size - overlap_size
        self.dilation = dilation

    def patchify_2d(self, x):
        x = x.unsqueeze(0)
        b, c, h, w = x.size()

        # patch up the images
        # (b, c, h, w) -> (b, c*patch_size, L)
        x = F.unfold(x,
                     kernel_size=self.patch_size,
                     stride=self.overlap_size,
                     dilation=self.dilation)

        # (b, c*patch_size, L) -> (b, nb_patches, width, height)
        x = x.transpose(2, 1).contiguous().view(b, -1, self.patch_size, self.patch_size)

        # reshape to have (b x patches, c, h, w)
        x = x.view(-1, c, self.patch_size, self.patch_size)

        x = x.squeeze(0)

        return x

    def __call__(self, x):
        if x.dim() == 3:
            return self.patchify_2d(x)
        else:
            raise NotImplementedError("patchify is only implemented for 2d images")


class BuildFromArgparse(object):
    @classmethod
    def from_argparse_args(cls, args, **kwargs):

        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)


def quantil_normalize(tensor, pmin=3, pmax=99.8, clip=4.,
                      eps=1e-20, dtype=np.float32, axis=None):
    mi = np.percentile(tensor, pmin, axis=axis, keepdims=True)
    ma = np.percentile(tensor, pmax, axis=axis, keepdims=True)

    if dtype is not None:
        tensor = tensor.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(tensor - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (tensor - mi) / (ma - mi + eps)

    if clip is not None:
        x = np.clip(x, -clip, clip)

    return x


class QuantileNormalize(Transform):
    """Percentile-based image normalization
       (adopted from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py)"""

    def __init__(self, pmin=0.6, pmax=99.8, clip=4.,
                 eps=1e-20, dtype=np.float32,
                 axis=None, **super_kwargs):
        """
        Parameters
        ----------
        pmin: float
            minimum percentile value. The pmin percentile value of the input tensor
            is mapped to 0.
        pmax: float
            maximum percentile value. The pmax percentile value of the input tensor
            is mapped to 1.
        clip: bool
            Clip all values outside of the percentile range to (0, 1)
        axis: int, tuple or None
            spatial dimensions considerered for the normalization
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super().__init__(**super_kwargs)
        self.pmin = pmin
        self.pmax = pmax
        self.clip = clip
        self.axis = axis
        self.dtype = dtype
        self.eps = eps

    def tensor_function(self, tensor):
        return quantil_normalize(tensor, pmin=self.pmin, pmax=self.pmax, clip=self.clip,
                                 axis=self.axis, dtype=self.dtype, eps=self.eps)


class QuantileNormalizeTorchTransform(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, pmin=3, pmax=99.8, clip=4.,
                 eps=1e-20, axis=None):
        self.pmin = pmin
        self.pmax = pmax
        self.clip = clip
        self.axis = axis
        self.eps = eps

    def __call__(self, sample):
        return quantil_normalize(sample, pmin=self.pmin, pmax=self.pmax, clip=self.clip,
                                 axis=self.axis, dtype=None, eps=self.eps).float()


def pre_channel(img, fun):
    if len(img.shape) == 3:
        return np.stack(tuple(fun(_) for _ in img), axis=0)
    else:
        return fun(img)


class Scale(Transform):
    """ Rescale patch of by constant factor"""

    def __init__(self, scale, **super_kwargs):
        super().__init__(**super_kwargs)
        self.scale = scale

    def batch_function(self, inp):

        image, segmentation = inp

        if self.scale != 1.:
            image = pre_channel(
                image,
                partial(rescale,
                        scale=self.scale,
                        order=3,
                        anti_aliasing=True))

            segmentation = pre_channel(
                segmentation,
                partial(rescale,
                        scale=self.scale,
                        order=0))

        return image.astype(np.float32), segmentation.astype(np.float32)


def import_by_string(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class SaveModelOnValidation(Callback):

    def __init__(self, run_segmentation=False, device='cpu'):
        self.run_segmentation = run_segmentation
        self.device = device
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        model_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                       os.pardir,
                                                       os.pardir,
                                                       "models"))
        os.makedirs(model_directory, exist_ok=True)
        if hasattr(pl_module, "mini_unet"):
            model_save_path = os.path.join(
                model_directory, f"mini_unet_{pl_module.global_step:08d}_{pl_module.local_rank:02}.torch")
            torch.save({"model_state_dict": pl_module.mini_unet.state_dict()}, model_save_path)

        if hasattr(pl_module, "maxi_unet"):
            model_save_path = os.path.join(
                model_directory, f"maxi_unet_{pl_module.global_step:08d}_{pl_module.local_rank:02}.torch")
            torch.save({"model_state_dict": pl_module.maxi_unet.state_dict()}, model_save_path)


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def sometimes(aug): return iaa.Sometimes(0.5, aug)


def get_augmentation_transform(simple=False, medium=False, seed=True):
    if simple:
        return iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(iaa.geometric.Rot90(k=ia.ALL, keep_size=False))
        ])
    if medium:
        return iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(iaa.geometric.Rot90(k=ia.ALL, keep_size=False)),
            sometimes(iaa.ElasticTransformation(alpha=(0., 10.), sigma=(3, 8))),
            iaa.SomeOf((0, 2),
                       [
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 1.0
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 1.),
                                          per_channel=False),  # add gaussian noise to images
                iaa.Multiply((0.8, 1.2), per_channel=False),
                iaa.LinearContrast((0.5, 2.0), per_channel=True),  # improve or worsen the contrast
            ], random_order=True)
        ])
    return iaa.Sequential([
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 50% of all images
        sometimes(iaa.ElasticTransformation(alpha=(0., 30.), sigma=(3, 8))),
        iaa.SomeOf((0, 2),
                   [
            iaa.geometric.ScaleX(scale=(1., 1.2), order=1, cval=0, mode='constant'),
            iaa.geometric.ScaleY(scale=(1., 1.2), order=1, cval=0, mode='constant'),
            iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
            iaa.Dropout((0.01, 0.1), per_channel=False),  # randomly remove up to 10% of the pixels
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 1.),
                                      per_channel=False),  # add gaussian noise to images
            iaa.Multiply((0.8, 1.2), per_channel=False),
            iaa.LinearContrast((0.5, 2.0), per_channel=True),  # improve or worsen the contrast
        ], random_order=True)
    ])


def cluster_embeddings(embeddings, eps=1, min_samples=5):
    b, c, h, w = embeddings.shape
    emb = embeddings.permute(0, 2, 3, 1)
    emb = emb.view(b, -1, c)
    clusters = cluster_embeddings_flat(emb, eps=eps, min_samples=min_samples)
    clusters = [c.reshape(h, w) for c in clusters]
    clusters = np.stack(clusters, axis=0)
    return clusters


def cluster_embeddings_flat(embeddings, eps=1, min_samples=5):

    batch_of_clusters = []
    # we assume input embeddings are in the form (b, p, c)
    start_label = 0
    for emb in embeddings:
        # emb.shape = (p, c)
        emb = emb.detach().cpu().numpy()
        clusters = DBSCAN(eps=eps,
                          min_samples=min_samples).fit_predict(emb)

        # check that clusters are consecutive
        a = np.unique(clusters)
        assert((a < 0).all() or a[a >= 0].max() + 1 == len(a[a >= 0]))

        # offset labels by previous maximum label
        clusters[clusters >= 0] += start_label
        # check consistency
        if len(batch_of_clusters):
            a = np.unique(batch_of_clusters[-1])
            b = np.unique(clusters)
            assert(not np.isin(a[a >= 0], b[b >= 0]).any())

        batch_of_clusters.append(clusters)

        start_label = clusters.max() + 1

    return batch_of_clusters


def remove_border(raw_0, raw_1, seg, min_size=4000):
    seg = seg.copy()
    for b in range(raw_0.shape[0]):
        local_max_0 = rank.maximum(raw_0[b], disk(1))
        local_min_0 = rank.minimum(raw_0[b], disk(1))
        local_max_1 = rank.maximum(raw_1[b], disk(1))
        local_min_1 = rank.minimum(raw_1[b], disk(1))
        m0 = local_max_0 == local_min_0
        m1 = local_max_1 == local_min_1
        mask = np.logical_and(m0, m1)
        mask_seg = label(mask)
        reg = regionprops(mask_seg)
        for props in reg:
            if props.area > min_size:
                seg[b, mask_seg == props.label] = 0
    return seg


def sizefilter(segmentation, min_size, filter_non_connected=True):

    if min_size == 0:
        return segmentation
    
    if filter_non_connected:
        filter_labels = measure.label(segmentation, background=0)
    else:
        filter_labels = segmentation
    ids, sizes = np.unique(filter_labels, return_counts=True)
    filter_ids = ids[sizes < min_size]
    mask = np.in1d(filter_labels, filter_ids).reshape(filter_labels.shape)
    segmentation[mask] = 0

    return segmentation


def sizefilter_batch(segmentation, min_size):

    if min_size == 0:
        return segmentation
    full_mask = []
    for b in range(segmentation.shape[0]):
        ids, sizes = np.unique(segmentation[b], return_counts=True)
        filter_ids = ids[sizes < min_size]
        full_mask.append(np.in1d(segmentation[b], filter_ids).reshape(segmentation[b].shape))
    full_mask = np.stack(full_mask, axis=0)
    segmentation = segmentation.copy()
    segmentation[full_mask] = 0
    return segmentation


def smooth_boundary_fn(segmentation):
    segmentation = segmentation.copy()
    initialfg = segmentation > 0
    mask = segmentation == 0
    m1 = segmentation[..., :-1] != segmentation[..., 1:]
    mask[..., :-1] += m1
    mask[..., 1:] += m1

    m2 = segmentation[..., :-1, :] != segmentation[..., 1:, :]
    mask[..., :-1, :] += m2
    mask[..., 1:, :] += m2
    segmentation[mask] = 0

    seeds = measure.label(segmentation, background=0)
    mask = segmentation > 1
    distance = ndimage.distance_transform_edt(mask) + 0.1 * np.random.rand(*segmentation.shape)
    segmentation[:] = watershed(-distance, seeds, mask=initialfg)

    return segmentation


def zarr_append(key, data, outzarr, attr=None):
    if key not in outzarr:
        outzarr.create_dataset(key, data=data, chunks=data.shape, compression="gzip")
        if attr is not None:
            outzarr[key].attrs[attr[0]] = attr[1]
    else:
        outzarr[key].append(data)


def zarr_insert(key, data, outzarr, attr=None):
    if key not in outzarr:
        outzarr.create_dataset(key, data=data, chunks=data.shape, compression="gzip")
        if attr is not None:
            outzarr[key].attrs[attr[0]] = attr[1]
    else:
        outzarr[key].append(data)

        # if self.global_step % 100 == 0:
        #     def log_hook_full(grad_input):
        #         outzarr = zarr.open(f"grad_{self.global_step}.zarr", "w")
        #         zarr_append("grad", grad_input.detach().cpu().numpy()[None], outzarr)
        #         zarr_append("x_aux", (x_aux).detach().cpu().numpy()[None], outzarr)
        #         zarr_append("target_aux", (target_aux).detach().cpu().numpy()[None], outzarr)
        #         zarr_append("network_prediction_aux", network_prediction_aux.detach().cpu().numpy()[None], outzarr)
        #         handle.remove()
        #     handle = network_prediction_aux.register_hook(log_hook_full)

def read_config_from_script(script_file, parser):
    parser

class CropAndSkipIgnore():
    def __init__(self, crop_fn, valid_crop=46):
        self.crop_fn = crop_fn
        self.valid_crop = valid_crop
        self.max_tries = 100
    
    def acceptable(self, seg):
        vc = self.valid_crop
        # check if crop contains a valid segment in the center
        return seg[:, vc:-vc, vc:-vc].max() > 0
    
    def __call__(self, image=None,
                 segmentation_maps=None):
        
        raw, gtseg = self.crop_fn(image=image,
                                  segmentation_maps=segmentation_maps)
        vc = self.valid_crop
        if gtseg[:, vc:-vc, vc:-vc].max() <= 0:
            print("no crop available", np.unique(gtseg))
        
        for _ in range(self.max_tries):
            if not self.acceptable(gtseg):
                raw, gtseg = self.crop_fn(image=image,
                                        segmentation_maps=segmentation_maps)
            else:
                break
        
        return raw, gtseg
