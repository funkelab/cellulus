import gunpowder as gp

from cellulus.configs.dataset_config import DatasetConfig


def predict(inference_config, model, dataset_meta_data) -> DatasetConfig:
    # TODO: predict and save directly to zarr
    # use gunpowder Scan and ZarrWrite

    # create new zarr container and dataset in current directory

    # return DatasetConfig of prediction dataset

    # get the dataset_config data out of inference_config
    dataset_config = inference_config.dataset_config

    # prediction crop size is the size of the scanned tiles to be provided to the model
    prediction_crop_size = inference_config.prediction_crop_size

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    num_dims = dataset_meta_data.num_dims
    num_channels = dataset_meta_data.num_dims

    model.eval()

    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi(
        (0,) * num_dims, (1, num_channels, *prediction_crop_size)
    )

    predict = gp.torch.Predict(model, inputs={"input": raw}, outputs={0: prediction})

    (
        gp.ZarrSource(dataset_config.container_path, {raw: dataset_config.dataset_name})
        + predict
        + gp.scan(scan_request)
        + gp.ZarrWrite(
            dataset_names={
                prediction: inference_config.output_dataset_config.dataset_name
            },
            output_dir=inference_config.output_dataset_config.container_path,
            output_filename=inference_config.output_filename,
        )
    )

    return DatasetConfig(
        container_path=inference_config.output_dataset_config.container_path,
        dataset_name=inference_config.output_dataset_config.dataset_name,
    )
    pass
