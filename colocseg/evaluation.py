import argparse
import copy
import inspect
import os
from numcodecs import gzip

import numpy as np
import skimage
import torch
from torch.nn.functional import sigmoid
import zarr
from PIL import Image
from pytorch_lightning.callbacks import Callback
from skimage.io import imsave
from sklearn.cluster import MeanShift
from deepcell_toolbox import metrics
import json
from stardist.matching import matching_dataset
from colocseg.inference import infer, affinity_segmentation, stardist_segmentation, cellpose_segmentation
from colocseg.utils import cluster_embeddings, label2color, zarr_append
from colocseg.visualizations import vis_anchor_embedding
from colocseg.metrics import segmentation_metric


def n(a):
    out = a - a.min()
    out /= out.max() + 1e-8
    return out


class AnchorSegmentationValidation(Callback):

    def __init__(self, run_segmentation=False, device='cpu'):
        self.run_segmentation = run_segmentation
        self.device = device
        self.metrics = metrics.Metrics('colocseg')
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        self.seg_scores = {}

    def predict_embedding(self, x, pl_module):

        with torch.no_grad():
            embedding_spatial = pl_module.forward(x.to(pl_module.device))

        return embedding_spatial

    def create_eval_dir(self, pl_module):
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))

        os.makedirs(eval_directory, exist_ok=True)
        return eval_directory

    def visualize_embeddings(self, embedding, x, filename):
        if embedding is None:
            return

        for b in range(len(embedding)):
            e = embedding[b].cpu().numpy()
            for c in range(0, embedding.shape[1], 2):
                raw = x[b, 0].cpu().numpy()
                if raw.shape[-1] != e.shape[-1]:
                    pd0 = (raw.shape[-1] - e.shape[-1]) // 2
                    pd1 = (raw.shape[-2] - e.shape[-2]) // 2
                    raw = raw[..., pd1:-pd1, pd0:-pd0]
                imsave(f"{filename}_{b}_{c}.jpg",
                       np.stack((n(e[c]), n(raw), n(e[c + 1])), axis=-1))

    def visualize_segmentation(self, seg, x, filename):
        colseg = label2color(seg).transpose(1, 2, 0)
        img = np.repeat(x[0, ..., None].cpu().numpy(), 3, axis=-1)
        blend = (colseg[..., :3] / 2) + img
        imsave(filename, blend)

    def visualize_embedding_vectors(self, embedding_relative, x, filename, downsample_factor=8):

        for b in range(len(embedding_relative)):
            e = embedding_relative[b].cpu().numpy()

            cx = np.arange(e.shape[-2], dtype=np.float32)
            cy = np.arange(e.shape[-1], dtype=np.float32)
            coords = np.meshgrid(cx, cy, copy=True)
            coords = np.stack(coords, axis=-1)
            e_transposed = np.transpose(e, (1, 2, 0))

            dsf = downsample_factor
            spatial_dims = coords.shape[-1]
            patch_coords = coords[dsf // 2::dsf, dsf // 2::dsf].reshape(-1, spatial_dims)
            patch_embedding = e_transposed[dsf // 2::dsf, dsf // 2::dsf, :spatial_dims].reshape(-1, spatial_dims)

            vis_anchor_embedding(patch_embedding,
                                 patch_coords,
                                 n(x[b].cpu().numpy()),
                                 grad=None,
                                 output_file=f"{filename}_{b}.jpg")

    def write_embedding_to_file(self, eval_data_file, embedding, embedding_relative, x, y):
        # write embedding and raw data to file
        z_array = zarr.open(eval_data_file, mode="w")
        for b in range(len(embedding_relative)):
            z_array.create_dataset(f"{b}/embedding", data=embedding_relative[b].cpu().numpy(), compression='gzip')
            z_array.create_dataset(f"{b}/embedding_abs", data=embedding.cpu().numpy()[b], compression='gzip')
            z_array.create_dataset(f"{b}/gt_segmentation", data=y.cpu().numpy()[b, None], compression='gzip')
            z_array.create_dataset(f"{b}/raw", data=x[b].cpu().numpy(), compression='gzip')
            threshold = skimage.filters.threshold_li(image=x[b].cpu().numpy())
            z_array.create_dataset(f"{b}/threshold_li", data=255 *
                                   (x[b].cpu().numpy() > threshold).astype(dtype=np.uint8), compression='gzip')

        return z_array

    def meanshift_segmentation(self, embedding, bandwidths):
        # Compute Meanshift Segmentation
        ms_segmentaiton = {}

        for i, bandwidth in enumerate(bandwidths):
            ms_segmentaiton[bandwidth] = []
            ms_seg = segment_with_meanshift(embedding, bandwidth)

            for b, seg in enumerate(ms_seg):
                ms_segmentaiton[bandwidth].append(seg)

        return ms_segmentaiton

    def visualizalize_segmentation_dict(self, segmentation_dict, x, filename):
        for k in segmentation_dict:
            for b, seg in enumerate(segmentation_dict[k]):
                print(f'writing {filename}_{b}_{k}.jpg')
                self.visualize_segmentation(seg, x[b], f'{filename}_{b}_{k}.jpg')

    def visualize_all(self, eval_directory, x, embedding_spatial, batch_idx, pl_module):
        vis_pointer_filename = f"{eval_directory}/pointer_embedding_{batch_idx}_{pl_module.local_rank}"
        vis_inst_embedding_filename = f"{eval_directory}/instance_embedding_{batch_idx}_{pl_module.local_rank}"
        vis_relembedding_filename = f"{eval_directory}/spatial_embedding_{batch_idx}_{pl_module.local_rank}"
        self.visualize_embedding_vectors(embedding_spatial, x, vis_pointer_filename)
        self.visualize_embeddings(embedding_spatial, x, vis_relembedding_filename)

    def full_evaluation(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        x, anchor_coordinates, refernce_coordinates, y = batch

        eval_directory = self.create_eval_dir(pl_module)

        embedding_spatial = self.predict_embedding(x,
                                                   pl_module)
        if batch_idx < 32:
            self.visualize_all(eval_directory, x, embedding_spatial, batch_idx, pl_module)

        cut = (y.shape[-1] - embedding_spatial.shape[-1]) // 2
        gt = y.cpu()[..., cut:-cut, cut:-cut].numpy()
        self.gt_segs.append(gt)

        eval_data_file = f"{eval_directory}/embedding_{pl_module.local_rank}.zarr"
        z_array = zarr.open(eval_data_file, mode="a")
        z_array.create_dataset(f"embedding/{batch_idx}", data=embedding_spatial.cpu().numpy(), compression='gzip')
        z_array.create_dataset(f"raw/{batch_idx}", data=x[..., cut:-cut, cut:-cut].cpu().numpy(), compression='gzip')

        embedding_spatial = embedding_spatial.cpu()
        embedding_spatial[:, 1] += torch.arange(embedding_spatial.shape[2])[None, :, None]
        embedding_spatial[:, 0] += torch.arange(embedding_spatial.shape[3])[None, None, :]

        if self.run_segmentation:
            ms_bandwidths = (8,)
            ms_segmentations = self.meanshift_segmentation(embedding_spatial, ms_bandwidths)

            for k in ms_segmentations:
                if f"meanshift_{k}" not in self.pred_segs:
                    self.pred_segs[f"meanshift_{k}"] = []
                self.pred_segs[f"meanshift_{k}"].append(np.stack(ms_segmentations[k]))

            for eps in [0.1, 0.25, 0.5, 1., 2, 4, 8]:
                clusters = cluster_embeddings(embedding_spatial, eps=1.)
                # add 1 to set the label -1 indicating that no cluster was found to label=0
                clusters += 1
                if f"dbscan_{eps}" not in self.pred_segs:
                    self.pred_segs[f"dbscan_{eps}"] = []
                self.pred_segs[f"dbscan_{eps}"].append(clusters)

            if batch_idx == 0:
                for i, seg in enumerate(clusters):
                    pred = label2color(seg).transpose(1, 2, 0)[..., :3]
                    gt_vis = label2color(gt[i]).transpose(1, 2, 0)[..., :3]
                    raw = x[i].cpu().repeat_interleave(2, dim=0)[:3, cut:-cut, cut:-cut].numpy().transpose(1, 2, 0)
                    imsave(f"{eval_directory}/instances_{batch_idx}_{i:03}.png",
                           np.concatenate((raw, pred, gt_vis), axis=0))
                    imsave(f"{eval_directory}/instances_{batch_idx}_{i:03}_raw.png", raw)
                    imsave(f"{eval_directory}/instances_{batch_idx}_{i:03}_pred.png", pred)
                    imsave(f"{eval_directory}/instances_{batch_idx}_{i:03}_gt.png", gt_vis)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.full_evaluation(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_start(self, trainer, pl_module):
        # reset segmentation values
        self.gt_segs = []
        self.pred_segs = {}

        eval_directory = self.create_eval_dir(pl_module)
        eval_data_file = f"{eval_directory}/embedding_{pl_module.local_rank}.zarr"
        zarr.open(eval_data_file, mode="w")

    def on_validation_end(self, trainer, pl_module):
        gt = np.concatenate(self.gt_segs, axis=0)

        eval_directory = self.create_eval_dir(pl_module)
        eval_data_file = f"{eval_directory}/embedding_{pl_module.local_rank}.zarr"
        z_array = zarr.open(eval_data_file, mode="a")
        z_array.create_dataset(f"gt", data=gt, compression='gzip')

        for k in self.pred_segs:
            predictions = np.concatenate(self.pred_segs[k], axis=0)
            z_array.create_dataset(f"predictions_{k}", data=predictions, compression='gzip')

            all_metrics = self.metrics.calc_object_stats(gt, predictions)
            all_metrics.to_csv(f"{eval_directory}/score_{k}.csv")


def segment_with_meanshift(embedding,
                           bandwidth,
                           mask=None,
                           reduction_probability=0.1,
                           cluster_all=False):
    ams = AnchorMeanshift(bandwidth,
                          reduction_probability=reduction_probability,
                          cluster_all=cluster_all)
    return ams(embedding, mask=mask) + 1


class AnchorMeanshift():
    def __init__(self, bandwidth, reduction_probability=0.1, cluster_all=False):
        self.ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
        self.reduction_probability = reduction_probability

    def compute_ms(self, X):
        if self.reduction_probability < 1.:
            X_reduced = X[np.random.rand(len(X)) < self.reduction_probability]
            ms_seg = self.ms.fit(X_reduced)
        else:
            ms_seg = self.ms.fit(X)

        ms_seg = self.ms.predict(X)

        return ms_seg

    def compute_masked_ms(self, embedding, mask=None):
        c, w, h = embedding.shape
        if mask is not None:
            assert len(mask.shape) == 2
            if mask.sum() == 0:
                return -1 * np.ones(mask.shape, dtype=np.int32)
            resh_emb = embedding.permute(1, 2, 0)[mask].view(-1, c)
        else:
            resh_emb = embedding.permute(1, 2, 0).view(w * h, c)
        resh_emb = resh_emb.contiguous().numpy()

        ms_seg = self.compute_ms(resh_emb)
        if mask is not None:
            ms_seg_spatial = -1 * np.ones(mask.shape, dtype=np.int32)
            ms_seg_spatial[mask] = ms_seg
            ms_seg = ms_seg_spatial
        else:
            ms_seg = ms_seg.reshape(w, h)
        return ms_seg

    def __call__(self, embedding, mask=None):
        segmentation = []
        for j in range(len(embedding)):
            mask_slice = mask[j] if mask is not None else None
            ms_seg = self.compute_masked_ms(embedding[j], mask=mask_slice)
            segmentation.append(ms_seg)

        return np.stack(segmentation)


class SegmentationValidation(Callback):

    def __init__(self, name, log_dir=None, min_size=30):
        super().__init__()
        self.min_size = min_size
        self.log_dir = log_dir
        self.name = name

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        self.scores = {}
        self.out_dir = self.create_eval_dir(pl_module)

    def create_eval_dir(self, pl_module):
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))

        os.makedirs(eval_directory, exist_ok=True)
        self.output_file = f"{eval_directory}/evaluation_{pl_module.local_rank}.zarr"
        # create a new zarr file (remove if it exists already)
        zarr.open(self.output_file, "w")
        return eval_directory

    def on_validation_end(self, trainer, pl_module):
        eval_score_file = f"{self.out_dir}/scores_{pl_module.local_rank}.csv"
        scores = evaluate_predicted_zarr(self.output_file, eval_score_file)
        pl_module.metrics = scores.to_dict()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        raise NotImplementedError()


class AffinitySegmentationValidation(SegmentationValidation):
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        network_prediction = infer(batch, pl_module, pl_module.valid_crop, sigmoid=True)

        raw = batch[0].cpu().numpy()
        mws_segmentation_supervised = affinity_segmentation(
            raw, network_prediction, min_size=self.min_size)

        outzarr = zarr.open(self.output_file, "a")
        zarr_append("raw", raw, outzarr)
        zarr_append("gt", batch[1].cpu().numpy(), outzarr)
        zarr_append("prediction", network_prediction, outzarr)
        zarr_append("segmentation", mws_segmentation_supervised, outzarr, attr=("name", self.name))


class StardistSegmentationValidation(SegmentationValidation):

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        network_prediction = infer(batch, pl_module, pl_module.valid_crop)

        raw = batch[0].cpu().numpy()
        network_prediction[:, 0] = torch.from_numpy(network_prediction[:, 0]).sigmoid().numpy()

        mws_segmentation_supervised = stardist_segmentation(
            raw, network_prediction, min_size=self.min_size)

        outzarr = zarr.open(self.output_file, "a")
        zarr_append("raw", raw, outzarr)
        zarr_append("gt", batch[1][None].cpu().numpy(), outzarr)
        if batch_idx == 0:
            zarr_append("prediction", network_prediction, outzarr)
        zarr_append("segmentation", mws_segmentation_supervised[None], outzarr, attr=("name", self.name))


class CellposeSegmentationValidation(SegmentationValidation):

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        network_prediction = infer(batch, pl_module, pl_module.valid_crop)

        raw = batch[0].cpu().numpy()
        segmentation_supervised = cellpose_segmentation(
            raw, network_prediction, min_size=self.min_size)

        outzarr = zarr.open(self.output_file, "a")
        zarr_append("raw", raw, outzarr)
        zarr_append("gt", batch[1][None].cpu().numpy(), outzarr)
        if batch_idx == 0:
            zarr_append("prediction", network_prediction, outzarr)
        zarr_append("segmentation", segmentation_supervised[None], outzarr, attr=("name", self.name))


def evaluate_predicted_zarr(zarr_file,
                            score_file_out,
                            gt_key="gt",
                            seg_key="segmentation",
                            hide_report=True,
                            progbar=False,
                            mask=False):
    # compute all metrics
    met = metrics.Metrics('colocseg')
    zarrin = zarr.open(zarr_file)
    if mask is None:
        gt = zarrin[gt_key]
        predictions = zarrin[seg_key]
    else:
        gt = zarrin[gt_key][mask]
        predictions = zarrin[seg_key][mask]

    if hide_report:
        met.print_object_report = lambda a: None

    object_metrics = met.calc_object_stats(gt, predictions, progbar=progbar)
    scores = segmentation_metric(gt, predictions, return_matches=True)

    with open(score_file_out[:-4] + "_matches.json", 'w') as file:
        json.dump(scores, file)

    object_metrics.to_csv(score_file_out)
    obj_mean = object_metrics.mean(axis=0)
    obj_mean["seg_w"] = scores["seg"]
    obj_mean["recall_w"] = scores["recall"]
    obj_mean["precision_w"] = scores["precision"]
    obj_mean["f1_w"] = scores["f1"]
    obj_mean["sum_gt_objects"] = scores["sum_gt_objects"]
    obj_mean["sum_pred_objects"] = scores["sum_pred_objects"]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for tau in taus:
        stats = matching_dataset(gt, predictions, thresh=tau, show_progress=False)._asdict()
        for k in stats:
            obj_mean[f"std_{k}_{tau:0.1f}"] = stats[k]

    return obj_mean
