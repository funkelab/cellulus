import torch

from cellulus.datasets import get_dataset
from cellulus.models import get_model

def infer(experiment_config):
    ...
    # basic structure should be as follows:

    # load model
    # load dataset to predict on
    # gunpowder scan across each file in predict dataset
    # save out result in user-specified location

    # we need a dataset for the data we're predicting on
    # We need to make a custom dataset, that inherits from zarr_dataset, but overrides the __setup_pipeline so that we get a gunpowder scan instead of a gunpowder random_location+augment

    inference_config = experiment_config.inference_config
    dataset_meta_data = DatasetMetaData(inference_config.predict_data_config)

    # we need the specs for the model that was used for training, right?
    model = get_model(
        in_channels=dataset_meta_data.get_num_channels(),
        out_channels=dataset_meta_data.get_num_channels(),
        num_fmaps=experiment_config.model_config.num_fmaps,
        fmap_inc_factor=experiment_config.model_config.fmap_inc_factor,
        features_in_last_layer=experiment_config.model_config.features_in_last_layer,
        downsampling_factors=experiment_config.model_config.downsampling_factors,
        num_spatial_dims=dataset_meta_data.get_num_spatial_dims(),
    )

    prediction = predict(inference_config.dataset_config, model)

    segmentation = segment(prediction, inference_config)

    post_process(segment, inference_config)

def load_model():
    pass
