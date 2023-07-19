import zarr
from scipy.ndimage import distance_transform_edt as dtedt
from tqdm import tqdm

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def post_process(inference_config: InferenceConfig) -> None:
    # filter small objects, erosion, etc.

    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    f = zarr.open(inference_config.segmentation_dataset_config.container_path)
    ds = f[inference_config.segmentation_dataset_config.dataset_name]

    # prepare the zarr dataset to write to
    f_postprocessed = zarr.open(
        inference_config.post_processed_dataset_config.container_path
    )
    ds_postprocessed = f_postprocessed.create_dataset(
        inference_config.post_processed_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            1,
            *ds.shape[-dataset_meta_data.num_spatial_dims:],
        ),
    )
    ds_postprocessed.attrs["resolution"] = (1,) * dataset_meta_data.num_dims
    ds_postprocessed.attrs["offset"] = (0,) * dataset_meta_data.num_dims

    for sample in tqdm(range(dataset_meta_data.num_samples)):
        segmentation = ds[sample, 0]
        distance_background = dtedt(ds[sample, 0] == 0)
        mask = distance_background < inference_config.growd
        distance_background = dtedt(mask)
        segmentation[distance_background < inference_config.threshold] = 0
        ds_postprocessed[sample, 0, ...] = segmentation
