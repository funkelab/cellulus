import numpy as np
import zarr
from skimage.filters import threshold_otsu
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData
from cellulus.utils.mean_shift import mean_shift_segmentation


def segment(inference_config: InferenceConfig) -> None:
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)

    f = zarr.open(inference_config.segmentation_dataset_config.container_path)
    ds = f[inference_config.segmentation_dataset_config.source_dataset_name]

    # prepare the zarr dataset to write to
    f_segmentation = zarr.open(
        inference_config.segmentation_dataset_config.container_path
    )
    ds_segmentation = f_segmentation.create_dataset(
        inference_config.segmentation_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            inference_config.num_bandwidths,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
    )

    ds_segmentation.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_segmentation.attrs["resolution"] = (1,) * dataset_meta_data.num_spatial_dims
    ds_segmentation.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    # prepare the zarr dataset to write to
    zarr.open(inference_config.segmentation_dataset_config.container_path)
    ds_binary_segmentation = f_segmentation.create_dataset(
        "binary_" + inference_config.segmentation_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
    )

    ds_binary_segmentation.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_binary_segmentation.attrs["resolution"] = (
        1,
    ) * dataset_meta_data.num_spatial_dims
    ds_binary_segmentation.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    for sample in tqdm(range(dataset_meta_data.num_samples)):
        embeddings = ds[sample]
        embeddings_std = embeddings[-1, ...]
        embeddings_mean = embeddings[
            np.newaxis, : dataset_meta_data.num_spatial_dims, ...
        ].copy()
        threshold = threshold_otsu(embeddings_std)
        ds_binary_segmentation[sample, 0, ...] = embeddings_std < threshold
        for bandwidth_factor in range(inference_config.num_bandwidths):
            segmentation = mean_shift_segmentation(
                embeddings_mean,
                embeddings_std,
                bandwidth=inference_config.bandwidth / (2**bandwidth_factor),
                min_size=inference_config.min_size,
                reduction_probability=inference_config.reduction_probability,
                threshold=threshold,
            )
            # Note that the line below is needed
            # because the embeddings_mean is modified
            # by mean_shift_segmentation
            embeddings_mean = embeddings[
                np.newaxis, : dataset_meta_data.num_spatial_dims, ...
            ].copy()
            ds_segmentation[
                sample,
                bandwidth_factor,
                ...,
            ] = segmentation
