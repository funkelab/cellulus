import gunpowder as gp
import torch

from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def predict(model: torch.nn.Module, inference_config: InferenceConfig) -> DatasetConfig:
    # get the dataset_config data out of inference_config
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData(dataset_config)

    # prediction crop size is the size of the scanned tiles to be provided to the model
    input_shape = gp.Coordinate(inference_config.crop_size)
    output_shape = gp.Coordinate(
        model(
            torch.zeros(
                (1, dataset_meta_data.num_channels, *input_shape), dtype=torch.float32
            )
        ).shape
    )
    context = (input_shape - output_shape) / 2

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_shape)
    scan_request.add(prediction, output_shape)

    predict = gp.torch.Predict(model, inputs={"input": raw}, outputs={0: prediction})

    pipeline = (
        # source zarr data from the dataset described in inference_config.dataset_config
        gp.ZarrSource(dataset_config.container_path, {raw: dataset_config.dataset_name})
        + gp.Pad(raw, context)
        # torch predict
        + predict
        # write to new output file
        + gp.ZarrWrite(
            dataset_names={
                prediction: inference_config.output_dataset_config.dataset_name
            },
            output_filename=inference_config.output_dataset_config.container_path,
        )
        # scan across prediction dataset
        + gp.Scan(scan_request)
    )

    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())

    # return the dataset config for the prediction zarr dataset
    return DatasetConfig(
        container_path=inference_config.output_dataset_config.container_path,
        dataset_name=inference_config.output_dataset_config.dataset_name,
    )
