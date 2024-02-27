import numpy as np
import zarr
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import distance_transform_edt as dtedt
from skimage.filters import threshold_otsu
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData
from cellulus.utils.misc import size_filter


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

    # remove halo
    if inference_config.post_processing == "cell":
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
    elif inference_config.post_processing == "nucleus":
        ds_raw = f[inference_config.dataset_config.dataset_name]
        for sample in tqdm(range(dataset_meta_data.num_samples)):
            for bandwidth_factor in range(inference_config.num_bandwidths):
                segmentation = ds[sample, bandwidth_factor]
                raw_image = ds_raw[sample, 0]
                ids = np.unique(segmentation)
                ids = ids[ids != 0]
                for id_ in ids:
                    raw_image_masked = raw_image[segmentation == id_]
                    threshold = threshold_otsu(raw_image_masked)
                    mask = (segmentation == id_) & (raw_image > threshold)
                    mask = binary_fill_holes(mask)
                    if dataset_meta_data.num_spatial_dims == 2:
                        y, x = np.where(mask)
                        ds_postprocessed[sample, bandwidth_factor, y, x] = id_
                    elif dataset_meta_data.num_spatial_dims == 3:
                        z, y, x = np.where(mask)
                        ds_postprocessed[sample, bandwidth_factor, z, y, x] = id_

            # remove non-connected components
            for bandwidth_factor in range(inference_config.num_bandwidths):
                ids = np.unique(ds_postprocessed[sample, bandwidth_factor])
                ids = ids[ids != 0]
                counter = np.max(ids) + 1
                for id_ in ids:
                    ma_id = ds_postprocessed[sample, bandwidth_factor] == id_
                    array, num_features = label(ma_id)
                    if num_features > 1:
                        ids_array = np.unique(array)
                        ids_array = ids_array[ids_array != 0]
                        for id_array in ids_array:
                            if dataset_meta_data.num_spatial_dims == 2:
                                y, x = np.where(array == id_array)
                                ds_postprocessed[
                                    sample, bandwidth_factor, y, x
                                ] = counter
                            elif dataset_meta_data.num_spatial_dims == 3:
                                z, y, x = np.where(array == id_array)
                                ds_postprocessed[
                                    sample, bandwidth_factor, z, y, x
                                ] = counter
                            counter += 1

    # size filter - remove small objects
    for sample in tqdm(range(dataset_meta_data.num_samples)):
        for bandwidth_factor in range(inference_config.num_bandwidths):
            ds_postprocessed[sample, bandwidth_factor, ...] = size_filter(
                ds_postprocessed[sample, bandwidth_factor], inference_config.min_size
            )
