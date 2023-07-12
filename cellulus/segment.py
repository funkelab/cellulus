import numpy as np
import zarr
from tqdm import tqdm

from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData
from cellulus.utils.mean_shift import mean_shift_segmentation


def segment(inference_config: InferenceConfig) -> DatasetConfig:
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    f = zarr.open(inference_config.prediction_dataset_config.container_path)
    ds = f[inference_config.prediction_dataset_config.dataset_name]

    # prepare the zarr dataset to write to
    f2 = zarr.open(inference_config.segmentation_dataset_config.container_path)
    ds2 = f2.create_dataset(
        inference_config.segmentation_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
    )
    ds2.attrs["resolution"] = (1,) * dataset_meta_data.num_dims
    ds2.attrs["offset"] = (0,) * dataset_meta_data.num_dims

    for sample in tqdm(range(dataset_meta_data.num_samples)):
        embeddings = ds[sample]
        embeddings_std = embeddings[-1, ...]
        embeddings_mean = embeddings[
            np.newaxis, : dataset_meta_data.num_spatial_dims, ...
        ]
        segmentation = mean_shift_segmentation(
            embeddings_mean,
            embeddings_std,
            bandwidth=inference_config.bandwidth,
            min_size=inference_config.min_size,
        )
        ds2[
            sample,
            0,
            ...,
        ] = segmentation

    # return the dataset config for the segmentation zarr dataset
    return DatasetConfig(
        container_path=inference_config.segmentation_dataset_config.container_path,
        dataset_name=inference_config.segmentation_dataset_config.dataset_name,
    )
