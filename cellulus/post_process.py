import zarr
from scipy.ndimage import distance_transform_edt as dtedt
from tqdm import tqdm

from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def post_process(inference_config: InferenceConfig) -> DatasetConfig:
    # filter small objects, erosion, etc.

    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    f = zarr.open(inference_config.segmentation_dataset_config.container_path)
    ds = f[inference_config.segmentation_dataset_config.dataset_name]

    # prepare the zarr dataset to write to
    f2 = zarr.open(inference_config.post_processed_dataset_config.container_path)
    ds2 = f2.create_dataset(
        inference_config.post_processed_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
    )
    ds2.attrs["resolution"] = (1,) * dataset_meta_data.num_dims
    ds2.attrs["offset"] = (0,) * dataset_meta_data.num_dims

    for s in tqdm(range(dataset_meta_data.num_samples)):
        segmentation = ds[s, 0]
        distance_background = dtedt(ds[s, 0] == 0)
        mask = distance_background < inference_config.growd
        distance_background = dtedt(mask)
        segmentation[distance_background < inference_config.threshold] = 0
        ds2[s, 0, ...] = segmentation

    # return the dataset config for the post processed zarr dataset
    return DatasetConfig(
        container_path=inference_config.post_processed_dataset_config.container_path,
        dataset_name=inference_config.post_processed_dataset_config.dataset_name,
    )
