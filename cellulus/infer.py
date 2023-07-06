import os

import torch

from cellulus.datasets.meta_data import DatasetMetaData
from cellulus.models import get_model
from cellulus.post_process import post_process
from cellulus.predict import predict
from cellulus.segment import segment

torch.backends.cudnn.benchmark = True


def infer(experiment_config):
    print(experiment_config)

    inference_config = experiment_config.inference_config
    model_config = experiment_config.model_config
    dataset_meta_data = DatasetMetaData(inference_config.dataset_config)

    # set model
    model = get_model(
        in_channels=dataset_meta_data.num_channels,
        out_channels=dataset_meta_data.num_spatial_dims,
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=model_config.downsampling_factors,
        num_spatial_dims=dataset_meta_data.num_spatial_dims,
    )
    model = model.cuda()

    # load checkpoint
    if os.path.exists(model_config.checkpoint):
        state = torch.load(model_config.checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        assert (
            False
        ), f"Model weights do not exist at this location :{model_config.checkpoint}!"

    # set in eval mode
    model.eval()

    # get predicted embeddings...
    prediction_dataset_config = predict(model, inference_config)
    # ...turn them into a segmentation...
    segmentation_dataset_config = segment(prediction_dataset_config, inference_config)
    # ...and post-process the segmentation
    post_processed_dataset_config = post_process(
        segmentation_dataset_config, inference_config
    )

    return (
        prediction_dataset_config,
        segmentation_dataset_config,
        post_processed_dataset_config,
    )
