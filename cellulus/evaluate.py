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
    SEG = 0
    n_ids = 0
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

        F1 = compute_F1(IoU)
        F1_list.append(F1)
        SEG_list.append(SEG_image / n_GTids_image)
        SEG += SEG_image
        n_ids += n_GTids_image
        print(
            f"For sample {sample}, F1 = {F1:.3f}, SEG = {SEG_image/n_GTids_image:.3f}"
        )
    print(f"The mean F1 score is {np.mean(F1_list)}")
    print(f"SEG for dataset  is {SEG/n_ids}")

    txt_file = "results.txt"
    with open(txt_file, "w") as f:
        f.writelines("file index, F1, SEG \n")
        f.writelines("+++++++++++++++++++++++++++++++++\n")
        for sample in range(dataset_meta_data.num_samples):
            f.writelines(
                f"{sample}, {F1_list[sample]:.05f}, {SEG_list[sample]:.05f} \n"
            )
        f.writelines("+++++++++++++++++++++++++++++++++\n")
        f.writelines(f"Avg. F1 is {np.mean(F1_list):.05f} \n")
        f.writelines(f"SEG for dataset is {SEG/n_ids:.05f} \n")


def compute_pairwise_IoU(prediction, groundtruth):
    prediction_ids = np.unique(prediction)[1:]
    groundtruth_ids = np.unique(groundtruth)[1:]
    IoU_table = np.zeros((len(prediction_ids), len(groundtruth_ids)), dtype=np.float32)
    IoG_table = np.zeros((len(prediction_ids), len(groundtruth_ids)), dtype=np.float32)
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
    return IoU_table, np.sum(IoU_table[IoG_table > 0.5]), len(groundtruth_ids)


def compute_F1(IoU_table, threshold=0.5):
    IoU_table_thresholded = IoU_table >= threshold
    FP = np.sum(np.sum(IoU_table_thresholded, axis=1) == 0)
    FN = np.sum(np.sum(IoU_table_thresholded, axis=0) == 0)
    TP = IoU_table.shape[1] - FN
    return 2 * TP / (2 * TP + FP + FN)
