# # Download Data

# In this notebook, we will download data and convert it to a zarr dataset. <br>

# For demonstration, we will use one image from the `Platynereis-Nuclei-CBG` dataset
# provided
# with [this](https://www.sciencedirect.com/science/article/pii/S1361841522001700)
# publication.

# Firstly, the `tif` raw images are downloaded to a directory indicated by `data_dir`.

from pathlib import Path

# +
import numpy as np
import tifffile
import zarr
from cellulus.utils.misc import extract_data
from csbdeep.utils import normalize
from skimage.transform import rescale
from tqdm import tqdm

# +
name = "3d-data-demo"
data_dir = "./data"

extract_data(
    zip_url="https://github.com/funkelab/cellulus/releases/download/v0.0.1-tag/3d-data-demo.zip",
    data_dir=data_dir,
    project_name=name,
)
# -

# Currently, `cellulus` expects that the images are isotropic (i.e. the
# voxel size along z dimension (which is usually undersampled) is the same
# as the voxel size alng the x and y dimensions).<br>
# This dataset has a step size of $2.031 \mu m$ in z and $0.406 \mu m$ along
# x and y dimensions, thus, the upsampling factor (which we refer to as
# `anisotropy` equals $2.031/0.406$. <br>
# These raw images are upsampled, intensity-normalized and appended in a list.
# Here, we use the percentile normalization technique.

anisotropy = 2.031 / 0.406

container_path = zarr.open(name + ".zarr")
subsets = ["train", "test"]
for subset in subsets:
    dataset_name = subset + "/raw"
    image_filenames = sorted((Path(data_dir) / name / subset).glob("*.tif"))
    print(f"Number of raw images in {subset} directory is {len(image_filenames)}")
    image_list = []

    for i in tqdm(range(len(image_filenames))):
        im = tifffile.imread(image_filenames[i]).astype(np.float32)
        im_normalized = normalize(im, 1, 99.8, axis=(0, 1, 2))
        im_rescaled = rescale(im_normalized, (anisotropy, 1.0, 1.0))
        image_list.append(im_rescaled[np.newaxis, ...])

    image_list = np.asarray(image_list)
    container_path[dataset_name] = image_list
    container_path[dataset_name].attrs["resolution"] = (1, 1, 1)
    container_path[dataset_name].attrs["axis_names"] = ("s", "c", "z", "y", "x")
