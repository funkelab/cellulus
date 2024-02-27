import gunpowder as gp
import torch
import zarr

from cellulus.configs.inference_config import InferenceConfig
from cellulus.datasets.meta_data import DatasetMetaData


def predict(
    model: torch.nn.Module,
    inference_config: InferenceConfig,
    normalization_factor: float,
) -> None:
    # get the dataset_config data out of inference_config
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)

    # set device
    device = torch.device(inference_config.device)

    model.set_infer(
        p_salt_pepper=inference_config.p_salt_pepper,
        num_infer_iterations=inference_config.num_infer_iterations,
        device=device,
    )

    # prediction crop size is the size of the scanned tiles to be provided to the model
    input_shape = gp.Coordinate(
        (1, dataset_meta_data.num_channels, *inference_config.crop_size)
    )

    output_shape = gp.Coordinate(
        model(
            torch.zeros(
                (1, dataset_meta_data.num_channels, *inference_config.crop_size),
                dtype=torch.float32,
            ).to(device)
        ).shape
    )

    # treat all dimensions as spatial, with a voxel size of 1
    voxel_size = (1,) * dataset_meta_data.num_dims
    raw_spec = gp.ArraySpec(voxel_size=voxel_size, interpolatable=True)

    input_size = gp.Coordinate(input_shape) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(output_shape) * gp.Coordinate(voxel_size)
    diff_size = input_size - output_size

    if dataset_meta_data.num_spatial_dims == 2:
        context = (0, 0, diff_size[2] // 2, diff_size[3] // 2)
    elif dataset_meta_data.num_spatial_dims == 3:
        context = (
            0,
            0,
            diff_size[2] // 2,
            diff_size[3] // 2,
            diff_size[4] // 2,
        )  # type: ignore

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    scan_request = gp.BatchRequest()
    if dataset_meta_data.num_spatial_dims == 2:
        scan_request[raw] = gp.Roi(
            (0, 0, -diff_size[2] // 2, -diff_size[3] // 2),
            (1, dataset_meta_data.num_channels, input_size[2], input_size[3]),
        )
        scan_request[prediction] = gp.Roi(
            (0, 0, 0, 0),
            (1, dataset_meta_data.num_spatial_dims + 1, output_size[2], output_size[3]),
        )
    elif dataset_meta_data.num_spatial_dims == 3:
        scan_request[raw] = gp.Roi(
            (0, 0, -diff_size[2] // 2, -diff_size[3] // 2, -diff_size[4] // 2),
            (
                1,
                dataset_meta_data.num_channels,
                input_size[2],
                input_size[3],
                input_size[4],
            ),
        )
        scan_request[prediction] = gp.Roi(
            (0, 0, 0, 0, 0),
            (
                1,
                dataset_meta_data.num_spatial_dims + 1,
                output_size[2],
                output_size[3],
                output_size[4],
            ),
        )

    predict = gp.torch.Predict(
        model,
        inputs={"raw": raw},
        outputs={0: prediction},
        array_specs={prediction: raw_spec},
    )

    # prepare the zarr dataset to write to
    f = zarr.open(inference_config.prediction_dataset_config.container_path)
    ds = f.create_dataset(
        inference_config.prediction_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            dataset_meta_data.num_spatial_dims + 1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=float,
    )

    pipeline = (
        gp.ZarrSource(
            dataset_config.container_path,
            {raw: dataset_config.dataset_name},
            {raw: gp.ArraySpec(voxel_size=voxel_size, interpolatable=True)},
        )
        + gp.Normalize(raw, factor=normalization_factor)
        + gp.Pad(raw, context, mode="reflect")
        + predict
        + gp.ZarrWrite(
            dataset_names={
                prediction: inference_config.prediction_dataset_config.dataset_name
            },
            output_filename=inference_config.prediction_dataset_config.container_path,
        )
        + gp.Scan(scan_request)
    )

    request = gp.BatchRequest()
    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(request)

    ds.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]

    ds.attrs["resolution"] = (1,) * dataset_meta_data.num_spatial_dims
    ds.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims
