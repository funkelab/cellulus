import numpy as np
import zarr
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def evaluate(inference_config: InferenceConfig) -> None:
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    f = zarr.open(inference_config.evaluation_dataset_config.container_path)
    ds = f[inference_config.evaluation_dataset_config.dataset_name]

    f_segmentation = zarr.open(
        inference_config.segmentation_dataset_config.container_path
    )
    ds_segmentation = f_segmentation[
        inference_config.segmentation_dataset_config.dataset_name
    ]

    F1_list = []
    SEG_list = []
    for sample in tqdm(range(dataset_meta_data.num_samples)):
        groundtruth = ds[sample, 0].astype(np.uint16)
        prediction = ds_segmentation[sample, 0].astype(np.uint16)
        IoU = compute_pairwise_IoU(prediction, groundtruth)

        F1 = compute_F1(IoU)
        SEG = compute_SEG(prediction, groundtruth)
        F1_list.append(F1)
        SEG_list.append(SEG)
        print(f"For sample {sample}, the F1 is {F1:.3f}, the SEG is {SEG:.3f}")
    print(f"The mean F1 score is {np.mean(F1_list)}")
    print(f"The mean SEG score is {np.mean(SEG_list)}")


def compute_pairwise_IoU(prediction, groundtruth):
    prediction_ids = np.unique(prediction)[1:]
    groundtruth_ids = np.unique(groundtruth)[1:]
    IoU_table = np.zeros((len(prediction_ids), len(groundtruth_ids)), dtype=np.float32)
    for j in range(len(prediction_ids)):
        for k in range(len(groundtruth_ids)):
            intersection = (prediction == prediction_ids[j]) & (
                groundtruth == groundtruth_ids[k]
            )
            union = (prediction == prediction_ids[j]) | (
                groundtruth == groundtruth_ids[k]
            )
            IoU_table[j, k] = np.sum(intersection) / np.sum(union)
    return IoU_table


def compute_F1(IoU_table, threshold=0.5):
    IoU_table_thresholded = IoU_table > threshold
    FP = np.sum(np.sum(IoU_table_thresholded, axis=1) == 0)
    FN = np.sum(np.sum(IoU_table_thresholded, axis=0) == 0)
    TP = IoU_table.shape[1] - FN
    return 2 * TP / (2 * TP + FP + FN)


def matching_overlap(psg, fractions=(0.5, 0.5)):
    afrac, bfrac = fractions
    tmp = np.sum(psg, axis=1, keepdims=True)
    m0 = np.where(tmp == 0, 0, psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp == 0, 0, psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype("bool")
    return matching


def compute_SEG(label, label_gt):
    psg = pixel_sharing_bipartite(label_gt, label)
    iou = intersection_over_union(psg)
    matching = matching_overlap(psg, fractions=(0.5, 0))
    matching[0, :] = False
    matching[:, 0] = False
    n_gt = len(set(np.unique(label_gt)) - {0})
    n_matched = iou[matching].sum()
    return n_matched / n_gt


def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg


def intersection_over_union(psg):
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)
