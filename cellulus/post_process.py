import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt as dtedt
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def post_process(inference_config: InferenceConfig) -> None:
    # filter small objects, erosion, etc.

    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)

    f = zarr.open(inference_config.post_processed_dataset_config.container_path)
    ds = f[inference_config.post_processed_dataset_config.secondary_dataset_name]

    # prepare the zarr dataset to write to
    f_postprocessed = zarr.open(
        inference_config.post_processed_dataset_config.container_path
    )
    ds_postprocessed = f_postprocessed.create_dataset(
        inference_config.post_processed_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            inference_config.num_bandwidths,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
    )

    ds_postprocessed.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_postprocessed.attrs["resolution"] = (1,) * dataset_meta_data.num_spatial_dims
    ds_postprocessed.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    for sample in tqdm(range(dataset_meta_data.num_samples)):
        # first instance label masks are expanded by `grow_distance`
        # next, expanded  instance label masks are shrunk by `shrink_distance`
        for bandwidth_factor in range(inference_config.num_bandwidths):
            segmentation = ds[sample, bandwidth_factor]
            distance_foreground = dtedt(segmentation == 0)
            expanded_mask = distance_foreground < inference_config.grow_distance
            distance_background = dtedt(expanded_mask)
            segmentation[distance_background < inference_config.shrink_distance] = 0
            ds_postprocessed[sample, bandwidth_factor, ...] = segmentation
