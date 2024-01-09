# # Infer using Trained Model

# In this notebook, we will use the `cellulus` model trained in the previous step to obtain instance segmentations.

import zipfile

import skimage
import torch
import zarr
from attrs import asdict
from cellulus.configs.dataset_config import DatasetConfig
from cellulus.configs.experiment_config import ExperimentConfig
from cellulus.configs.inference_config import InferenceConfig
from cellulus.configs.model_config import ModelConfig
from cellulus.infer import infer
from cellulus.utils.misc import visualize_2d

# ## Specify config values for datasets

# We again specify `name` of the zarr container, and `dataset_name` which identifies the path to the raw image data, which needs to be segmented.

name = "2d-data-demo"
dataset_name = "train/raw"

# We initialize the `dataset_config` which relates to the raw image data, `prediction_dataset_config` which relates to the per-pixel embeddings and the uncertainty, the `segmentation_dataset_config` which relates to the segmentations post the mean-shift clustering and the `post_processed_config` which relates to the segmentations after some post-processing.

dataset_config = DatasetConfig(container_path=name + ".zarr", dataset_name=dataset_name)
prediction_dataset_config = DatasetConfig(
    container_path=name + ".zarr", dataset_name="embeddings"
)
segmentation_dataset_config = DatasetConfig(
    container_path=name + ".zarr",
    dataset_name="segmentation",
    secondary_dataset_name="embeddings",
)
post_processed_dataset_config = DatasetConfig(
    container_path=name + ".zarr",
    dataset_name="post_processed_segmentation",
    secondary_dataset_name="segmentation",
)

# ## Specify config values for the model

# We must also specify the `num_fmaps`, `fmap_inc_factor` (use same values as in the training step) and set `checkpoint` equal to `models/best_loss.pth` (<i>best</i> in terms of the lowest loss obtained).

# Here, we download a pretrained model trained by us for `5e3` iterations. <br>
# But please comment the next cell to use <i>your own</i> trained model, which should be available in the `models` directory.

torch.hub.download_url_to_file(
    url="https://github.com/funkelab/cellulus/releases/download/v0.0.1-tag/2d-demo-model.zip",
    dst="pretrained_model",
    progress=True,
)
with zipfile.ZipFile("pretrained_model", "r") as zip_ref:
    zip_ref.extractall("")

num_fmaps = 24
fmap_inc_factor = 3
checkpoint = "models/best_loss.pth"

model_config = ModelConfig(
    num_fmaps=num_fmaps, fmap_inc_factor=fmap_inc_factor, checkpoint=checkpoint
)

# ## Initialize `inference_config`

# Then, we specify inference-specific parameters such as the `device`, which indicates the actual device to run the inference on.
# <br> The device could be set equal to `cuda:n` (where `n` is the index of the GPU, for e.g. `cuda:0`), `cpu` or `mps`.

device = "mps"

# We initialize the `inference_config` which contains our `embeddings_dataset_config`, `segmentation_dataset_config` and `post_processed_dataset_config`.

inference_config = InferenceConfig(
    dataset_config=asdict(dataset_config),
    prediction_dataset_config=asdict(prediction_dataset_config),
    segmentation_dataset_config=asdict(segmentation_dataset_config),
    post_processed_dataset_config=asdict(post_processed_dataset_config),
    device=device,
)

# ## Initialize `experiment_config`

# Lastly we initialize the `experiment_config` which contains the `inference_config` and `model_config` initialized above.

experiment_config = ExperimentConfig(
    inference_config=asdict(inference_config), model_config=asdict(model_config)
)

# Now we are ready to start the inference!! <br>
# (This takes around 7 minutes on a Mac Book Pro with an Apple M2 Max chip).

infer(experiment_config)

# ## Inspect predictions

# Let's look at some of the predicted embeddings. <br>
# Change the value of `index` below to look at the raw image (left), x-offset (bottom-left), y-offset (bottom-right) and uncertainty of the embedding (top-right).

# +
index = 0

f = zarr.open(name + ".zarr")
ds = f["train/raw"]
ds2 = f["centered_embeddings"]

image = ds[index, 0]
embedding = ds2[index]

visualize_2d(
    image,
    top_right=embedding[-1],
    bottom_left=embedding[0],
    bottom_right=embedding[1],
    top_right_label="STD_DEV",
    bottom_left_label="OFFSET_X",
    bottom_right_label="OFFSET_Y",
)
# -

# As you can see the magnitude of the uncertainty of the embedding (top-right) is <i>low</i> for most of the foreground cells. <br> This enables extraction of the foreground, which is eventually clustered into individual instances.

# +
index = 0

f = zarr.open(name + ".zarr")
ds = f["train/raw"]
ds2 = f["segmentation"]
ds3 = f["post_processed_segmentation"]

visualize_2d(
    image,
    top_right=embedding[-1] < skimage.filters.threshold_otsu(embedding[-1]),
    bottom_left=ds2[0, 0],
    bottom_right=ds3[0, 0],
    top_right_label="THRESHOLDED F.G.",
    bottom_left_label="SEGMENTATION",
    bottom_right_label="POSTPROCESSED",
    top_right_cmap="gray",
    bottom_left_cmap="nipy_spectral",
    bottom_right_cmap="nipy_spectral",
)
