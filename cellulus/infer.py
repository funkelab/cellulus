import os

import torch

from cellulus.datasets import get_dataset
from cellulus.models import get_model
from cellulus.predict import predict

# from cellulus.segment import segment
torch.backends.cudnn.benchmark = True


def infer(experiment_config):
    print(experiment_config)

    inference_config = experiment_config.inference_config
    model_config = experiment_config.model_config

    # create test_dataset
    test_dataset = get_dataset(
        mode="eval",
        dataset_config=inference_config.inference_data_config,
        crop_size=inference_config.crop_size,
    )

    # create test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False
    )

    # set model
    model = get_model(
        in_channels=test_dataset.get_num_channels(),
        out_channels=test_dataset.get_num_channels(),
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

    # create test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False
    )

    # get predicted embeddings
    predict(test_dataloader, model, inference_config)

    # segment(prediction, inference_config)

    # post_process(segment, inference_config)
