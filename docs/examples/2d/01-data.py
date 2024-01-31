# # Download Data

# In this notebook, we will download data and convert it to a zarr dataset. <br>
# This tutorial was written by <i>Henry Westmacott</i> and <i>Manan Lalit</i>.

# For demonstration, we will use a subset of images of `Fluo-N2DL-HeLa` available on the [Cell Tracking Challenge](http://celltrackingchallenge.net/2d-datasets/) webpage.

# Firstly, the `tif` raw images are downloaded to a directory indicated by `data_dir`.

from pathlib import Path

import numpy as np
import tifffile
import zarr
from cellulus.utils.misc import extract_data
from csbdeep.utils import normalize
from tqdm import tqdm

# +
name = "2d-data-demo"
data_dir = "./data"

extract_data(
    zip_url="https://github.com/funkelab/cellulus/releases/download/v0.0.1-tag/2d-data-demo.zip",
    data_dir=data_dir,
    project_name=name,
)
# -

# Next, these raw images are intensity-normalized and appended in a list. Here, we use the percentile normalization technique.

# +
container_path = zarr.open(name + ".zarr")
dataset_name = "train/raw"
image_filenames = sorted((Path(data_dir) / name / "images").glob("*.tif"))
print(f"Number of raw images is {len(image_filenames)}")
image_list = []

for i in tqdm(range(len(image_filenames))):
    im = normalize(
        tifffile.imread(image_filenames[i]).astype(np.float32),
        pmin=1,
        pmax=99.8,
        axis=(0, 1),
    )
    image_list.append(im[np.newaxis, ...])

image_list = np.asarray(image_list)
# -

# Lastly, the zarr dataset is populated, the axis names and resolution is specified.

container_path[dataset_name] = image_list
container_path[dataset_name].attrs["resolution"] = (1, 1)
container_path[dataset_name].attrs["axis_names"] = ("s", "c", "y", "x")
