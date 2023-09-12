import numpy as np
import zarr
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def evaluate(inference_config: InferenceConfig) -> None:
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)

    f = zarr.open(inference_config.evaluation_dataset_config.container_path)
    ds = f[inference_config.evaluation_dataset_config.dataset_name]

    f_segmentation = zarr.open(
        inference_config.post_processed_dataset_config.container_path
    )
    ds_segmentation = f_segmentation[
        inference_config.post_processed_dataset_config.dataset_name
    ]

    F1_list = []
    SEG_list = []
    SEG_dataset = 0
    TP = 0
    FP = 0
    FN = 0
    n_ids_dataset = 0
    for sample in tqdm(range(dataset_meta_data.num_samples)):
        if np.any(ds[sample, 0] - ds[sample, 0].astype(np.uint16)):
            mapping = {v: k for k, v in enumerate(np.unique(ds[sample, 0]))}
            u, inv = np.unique(ds[sample, 0], return_inverse=True)
            Y1 = np.array([mapping[x] for x in u])[inv].reshape(ds[sample, 0].shape)
            groundtruth = Y1.astype(np.uint16)
        else:
            groundtruth = ds[sample, 0].astype(np.uint16)
        prediction = ds_segmentation[sample, 0].astype(np.uint16)
        IoU, SEG_image, n_GTids_image = compute_pairwise_IoU(prediction, groundtruth)
        F1_image, TP_image, FP_image, FN_image = compute_F1(IoU)
        F1_list.append(F1_image)
        SEG_list.append(SEG_image / n_GTids_image)
        SEG_dataset += SEG_image
        n_ids_dataset += n_GTids_image
        TP += TP_image
        FP += FP_image
        FN += FN_image
        print(
            f"For sample {sample}, F1={F1_image:.3f}, SEG={SEG_image/n_GTids_image:.3f}"
        )
    print(f"The mean F1 score is {np.mean(F1_list)}")
    print(f"The mean SEG score is {np.mean(SEG_list)}")

    print(f"F1 for dataset  is {2*TP/(2*TP+FP+FN)}")
    print(f"SEG for dataset  is {SEG_dataset/n_ids_dataset}")

    txt_file = "results.txt"
    with open(txt_file, "w") as f:
        f.writelines("file index, F1, SEG \n")
        f.writelines("+++++++++++++++++++++++++++++++++\n")
        for sample in range(dataset_meta_data.num_samples):
            f.writelines(
                f"{sample}, {F1_list[sample]:.05f}, {SEG_list[sample]:.05f} \n"
            )
        f.writelines("+++++++++++++++++++++++++++++++++\n")
        f.writelines(f"Avg. F1 (averaged per sample) is {np.mean(F1_list):.05f} \n")
        f.writelines(f"Avg. SEG (averaged per sample) is {np.mean(SEG_list):.05f} \n")
        f.writelines(f"F1 for complete dataset is {2*TP/(2*TP+FP+FN):.05f} \n")
        f.writelines(f"SEG for complete dataset is {SEG_dataset/n_ids_dataset:.05f} \n")


def compute_pairwise_IoU(prediction, groundtruth):
    prediction_ids = np.unique(prediction)
    prediction_ids = prediction_ids[prediction_ids != 0]  # ignore background
    groundtruth_ids = np.unique(groundtruth)
    groundtruth_ids = groundtruth_ids[groundtruth_ids != 0]  # ignore background

    IoU_table = np.zeros((len(prediction_ids), len(groundtruth_ids)), dtype=float)
    IoG_table = np.zeros((len(prediction_ids), len(groundtruth_ids)), dtype=float)
    for j in range(len(prediction_ids)):
        for k in range(len(groundtruth_ids)):
            intersection = (prediction == prediction_ids[j]) & (
                groundtruth == groundtruth_ids[k]
            )
            union = (prediction == prediction_ids[j]) | (
                groundtruth == groundtruth_ids[k]
            )
            IoU_table[j, k] = np.sum(intersection) / np.sum(union)
            IoG_table[j, k] = np.sum(intersection) / np.sum(
                groundtruth == groundtruth_ids[k]
            )
    # Note for SEG, we consider it a match if it is strictly
    # greater than `0.5` IoU
    return IoU_table, np.sum(IoU_table[IoG_table > 0.5]), len(groundtruth_ids)


def compute_F1(IoU_table, threshold=0.5):
    IoU_table_thresholded = IoU_table >= threshold
    FP = np.sum(np.sum(IoU_table_thresholded, axis=1) == 0)
    FN = np.sum(np.sum(IoU_table_thresholded, axis=0) == 0)
    TP = IoU_table.shape[1] - FN
    return 2 * TP / (2 * TP + FP + FN), TP, FP, FN
