import gunpowder as gp
import torch
import zarr

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def predict(model: torch.nn.Module, inference_config: InferenceConfig) -> None:
    # get the dataset_config data out of inference_config
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    voxel_size = gp.Coordinate((1,) * dataset_meta_data.num_dims)

    model.set_infer(
        p_salt_pepper=inference_config.p_salt_pepper,
        num_infer_iterations=inference_config.num_infer_iterations,
    )

    # prediction crop size is the size of the scanned tiles to be provided to the model
    input_shape = gp.Coordinate(
        1, dataset_meta_data.num_channels, *inference_config.crop_size
    )
    output_shape = gp.Coordinate(
        model(
            torch.zeros(
                (1, dataset_meta_data.num_channels, *inference_config.crop_size),
                dtype=torch.float32,
            ).cuda()
        ).shape
    )
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) / 2

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(prediction, output_size)

    predict = gp.torch.Predict(
        model,
        inputs={"raw": raw},
        outputs={0: prediction},
        array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
    )

    # prepare the zarr dataset to write to
    f = zarr.open(inference_config.prediction_dataset_config.container_path)
    ds = f.create_dataset(
        inference_config.prediction_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            dataset_meta_data.num_channels + 1,
            *dataset_meta_data.spatial_array,
        ),
    )
    ds.attrs["resolution"] = (1,) * dataset_meta_data.num_dims
    ds.attrs["offset"] = (0,) * dataset_meta_data.num_dims

    pipeline = (
        gp.ZarrSource(
            dataset_config.container_path,
            {raw: dataset_config.dataset_name},
            {raw: gp.ArraySpec(voxel_size=voxel_size)},
        )
        + gp.Pad(raw, context)
        + predict
        + gp.ZarrWrite(
            dataset_names={
                prediction: inference_config.prediction_dataset_config.dataset_name
            },
            output_filename=inference_config.prediction_dataset_config.container_path,
        )
        + gp.Scan(scan_request)
    )

    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
