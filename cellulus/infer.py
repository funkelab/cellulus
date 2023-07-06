import os

import torch

from cellulus.models import get_model
from cellulus.predict import predict
from cellulus.segment import segment
from cellulus.post_process import post_process

torch.backends.cudnn.benchmark = True


def infer(experiment_config):
    print(experiment_config)

    inference_config = experiment_config.inference_config
    model_config = experiment_config.model_config

    # set model
    model = get_model(
        in_channels=test_dataset.get_num_channels(),
        out_channels=test_dataset.get_num_spatial_dims(),
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=model_config.downsampling_factors,
        num_spatial_dims=test_dataset.get_num_spatial_dims(),
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
