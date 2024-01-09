# # Train Model

# In this notebook, we will train a `cellulus` model.

from attrs import asdict
from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.experiment_config import ExperimentConfig
from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig

# ## Specify config values for dataset

# In the next cell, we specify the name of the zarr container and the dataset within it from which data would be read.

name = "2d-data-demo"
dataset_name = "train/raw"

train_data_config = DatasetConfig(
    container_path=name + ".zarr", dataset_name=dataset_name
)

# ## Specify config values for model

# In the next cell, we specify the number of feature maps (`num_fmaps`) in the first layer in our model. <br>
# Additionally, we specify `fmap_inc_factor`, which indicates by how much the number of feature maps increase between adjacent layers.

num_fmaps = 24
fmap_inc_factor = 3

model_config = ModelConfig(num_fmaps=num_fmaps, fmap_inc_factor=fmap_inc_factor)

# ## Specify config values for the training process

# Then, we specify training-specific parameters such as the `device`, which indicates the actual device to run the training on.
# <br> The device could be set equal to `cuda:n` (where `n` is the index of the GPU, for e.g. `cuda:0`), `cpu` or `mps`. <br>
# We set the `max_iterations` equal to `5e3` for demonstration purposes. <br>(This takes around 20 minutes on a Mac Book Pro with an Apple M2 Max chip).

device = "mps"
max_iterations = 5e3

train_config = TrainConfig(
    train_data_config=asdict(train_data_config),
    device=device,
    max_iterations=max_iterations,
)

# Next, we initialize the experiment config which puts together the config objects (`train_config` and `model_config`) which we defined above.

experiment_config = ExperimentConfig(
    train_config=asdict(train_config), model_config=asdict(model_config)
)

# Now we can begin the training! <br>
# Uncomment the next two lines to train the model.

# +
# from cellulus.train import train
# train(experiment_config)
