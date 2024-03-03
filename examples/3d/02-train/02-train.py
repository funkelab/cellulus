# # Train Model

# In this notebook, we will train a `cellulus` model.

from attrs import asdict
from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.experiment_config import ExperimentConfig
from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig

# ## Specify config values for dataset

# In the next cell, we specify the name of the zarr container and the dataset
# within it from which data would be read.

name = "3d-data-demo"
dataset_name = "train/raw"

train_data_config = DatasetConfig(
    container_path=name + ".zarr", dataset_name=dataset_name
)

# ## Specify config values for model

# In the next cell, we specify the number of feature maps (`num_fmaps`) in the
# first layer in our model. <br>
# Additionally, we specify `fmap_inc_factor` and `downsampling_factors`, which
# indicates by how much the number of feature maps increase between adjacent
# layers, and how much the spatial extents of the image gets downsampled between
# adjacent layers respectively.

num_fmaps = 24
fmap_inc_factor = 3
downsampling_factors = [
    [2, 2, 2],
]

model_config = ModelConfig(
    num_fmaps=num_fmaps,
    fmap_inc_factor=fmap_inc_factor,
    downsampling_factors=downsampling_factors,
)

# ## Specify config values for the training process

# Then, we specify training-specific parameters such as the `device`, which
# indicates the actual device to run the training on.
# We also specify the `crop_size`. Mini - batches of crops are shown to the model
# during training.
# <br> The device could be set equal to `cuda:n` (where `n` is the index of
# the GPU, for e.g. `cuda:0`) or `cpu`. <br>
# We set the `max_iterations` equal to `5000` for demonstration purposes.

device = "cuda:0"
max_iterations = 5000
crop_size = [80, 80, 80]

train_config = TrainConfig(
    train_data_config=asdict(train_data_config),
    device=device,
    max_iterations=max_iterations,
    crop_size=crop_size,
)

# Next, we initialize the experiment config which puts together the config
# objects (`train_config` and `model_config`) which we defined above.

experiment_config = ExperimentConfig(
    train_config=asdict(train_config),
    model_config=asdict(model_config),
    normalization_factor=1.0,  # since we already normalized in previous notebook
)

# Now we can begin the training! <br>
# Uncomment the next two lines to train the model.

# +
# from cellulus.train import train
# train(experiment_config)
# -
