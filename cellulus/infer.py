import os

import numpy as np
import torch

from cellulus.datasets.meta_data import DatasetMetaData
from cellulus.evaluate import evaluate
from cellulus.models import get_model
from cellulus.post_process import post_process
from cellulus.predict import predict
from cellulus.segment import segment

torch.backends.cudnn.benchmark = True


def infer(experiment_config):
    print(experiment_config)

    inference_config = experiment_config.inference_config
    normalization_factor = experiment_config.normalization_factor

    model_config = experiment_config.model_config

    dataset_meta_data = DatasetMetaData.from_dataset_config(
        inference_config.dataset_config
    )

    if inference_config.bandwidth is None:
        inference_config.bandwidth = 0.5 * experiment_config.object_size

    if inference_config.min_size is None:
        if dataset_meta_data.num_spatial_dims == 2:
            inference_config.min_size = int(
                0.1 * np.pi * (experiment_config.object_size**2) / 4
            )
        elif dataset_meta_data.num_spatial_dims == 3:
            inference_config.min_size = int(
                0.1 * 4.0 / 3.0 * np.pi * (experiment_config.object_size**3)
            )
    # set model
    model = get_model(
        in_channels=dataset_meta_data.num_channels,
        out_channels=dataset_meta_data.num_spatial_dims,
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=[
            tuple(factor) for factor in model_config.downsampling_factors
        ],
        num_spatial_dims=dataset_meta_data.num_spatial_dims,
    )
    # set device
    device = torch.device(inference_config.device)

    model = model.to(device)

    # load checkpoint
    if os.path.exists(model_config.checkpoint):
        state = torch.load(model_config.checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        assert (
            False
        ), f"Model weights do not exist at this location :{model_config.checkpoint}!"

    # set in eval mode
    model.eval()

    # get predicted embeddings...
    if inference_config.prediction_dataset_config is not None:
        predict(model, inference_config, normalization_factor)
    # ...turn them into a segmentation...
    if inference_config.segmentation_dataset_config is not None:
        segment(inference_config)
    # ...and post-process the segmentation
    if inference_config.post_processed_dataset_config is not None:
        post_process(inference_config)
    # ...and evaluate if ground-truth exists
    if inference_config.evaluation_dataset_config is not None:
        evaluate(inference_config)
