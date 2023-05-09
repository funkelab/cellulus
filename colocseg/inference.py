import numpy as np
import stardist
import torch
import torch.nn.functional as F
import cellpose
from affogato.segmentation import compute_mws_segmentation

from cellpose.dynamics import compute_masks
from colocseg.transforms import get_offsets
from colocseg.utils import remove_border, sizefilter, sizefilter_batch


def crop_to_fit(tensor, target_shape):
    remaining_cut_1_l = (tensor.shape[-1] - target_shape[-1]) // 2
    remaining_cut_1_r = (tensor.shape[-1] - target_shape[-1]) - remaining_cut_1_l
    remaining_cut_2_l = (tensor.shape[-2] - target_shape[-2]) // 2
    remaining_cut_2_r = (tensor.shape[-2] - target_shape[-2]) - remaining_cut_2_l
    return tensor[..., remaining_cut_2_l:-remaining_cut_2_r, remaining_cut_1_l:-remaining_cut_1_r]


def mws_segmentation(affinities, mask, seperating_channel=2, offsets=None, strides=(4, 4)):

    offsets = get_offsets() if offsets is None else offsets
    attractive_repulsive_weights = affinities.copy()
    attractive_repulsive_weights[:, :seperating_channel, ...] *= -1
    attractive_repulsive_weights[:, :seperating_channel, ...] += +1
    predicted_segmentation = []

    for i in range(attractive_repulsive_weights.shape[0]):
        predicted_segmentation.append(compute_mws_segmentation(
            attractive_repulsive_weights[i],
            offsets,
            seperating_channel,
            strides=strides,
            randomize_strides=True,
            mask=mask[i]))
    return np.stack(predicted_segmentation)


def infer(batch, model, valid_crop, sigmoid=False):

    x, y = batch
    p2d = (valid_crop * 2, valid_crop * 2, valid_crop * 2, valid_crop * 2)
    x_padded = F.pad(x, p2d, mode='constant')
    network_prediction = model.forward(x_padded)[-1]
    network_prediction = crop_to_fit(network_prediction, y.shape)
    if sigmoid:
        network_prediction.sigmoid_()

    return network_prediction.cpu().numpy()


def affinity_segmentation(raw, network_prediction, min_size):

    segmentation = mws_segmentation(
        network_prediction[:, 1:],
        mask=network_prediction[:, 0] > 0.5)

    # the validation set contains a large amount of padding
    # remove all segments outside of valid data area
    segmentation = remove_border(raw[:, 0], raw[:, 1], segmentation)
    segmentation = sizefilter_batch(segmentation, min_size=min_size)

    return segmentation.astype(np.int32)


def stardist_instances_from_prediction(dist, prob, prob_thresh=0.486166, nms_thresh=0.5, grid=(1, 1)):
    points, probi, disti = stardist.nms.non_maximum_suppression(dist, prob, grid=grid,
                                                                prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    img_shape = prob.shape
    return stardist.geometry.polygons_to_label(disti, points, prob=probi, shape=img_shape)


def batched_stardist_inference(predictions, prob_thresh=0.486166, nms_thresh=0.5, grid=(1, 1)):

    predicted_segmentation = []
    for i in range(predictions.shape[0]):
        dist = np.transpose(predictions[i, 1:], (1, 2, 0))
        prob = predictions[i, 0]
        assert dist.shape[-1] == 16, f"Unexpected stardist channels, dist.shape={dist.shape}"
        predicted_segmentation.append(
            stardist_instances_from_prediction(
                dist,
                prob, grid=grid,
                prob_thresh=prob_thresh, nms_thresh=nms_thresh))

    return np.stack(predicted_segmentation)


def stardist_segmentation(raw, network_prediction, min_size, prob_thresh=0.486166):
    segmentation = batched_stardist_inference(network_prediction, prob_thresh=prob_thresh)
    # the validation set contains a large amount of padding
    # remove all segments outside of valid data area
    segmentation = remove_border(raw[:, 0], raw[:, 1], segmentation)
    segmentation = sizefilter_batch(segmentation, min_size=min_size)

    return segmentation.astype(np.int32)

def cellpose_instances_from_prediction(network_prediction):
    flow = network_prediction[:2]
    distance_field = network_prediction[2]    
    try:
        seg, _, _ = compute_masks(flow, distance_field, mask_threshold=1.0, flow_threshold=None, min_size=10)
    except ValueError:
        seg = np.zeros(distance_field.shape)
    return seg


def batched_celpose_inference(predictions):
    predicted_segmentation = []
    for i in range(predictions.shape[0]):
        predicted_segmentation.append(
            cellpose_instances_from_prediction(predictions[i]))
    return np.stack(predicted_segmentation)


def cellpose_segmentation(raw, network_prediction, min_size):

    segmentation = batched_celpose_inference(network_prediction)
    # the validation set contains a large amount of padding
    # remove all segments outside of valid data area
    segmentation = remove_border(raw[:, 0], raw[:, 1], segmentation)
    segmentation = sizefilter_batch(segmentation, min_size=min_size)

    return segmentation.astype(np.int32)
