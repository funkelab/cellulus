import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def size_filter(segmentation, min_size, filter_non_connected=True):
    if min_size == 0:
        return segmentation

    if filter_non_connected:
        filter_labels = measure.label(segmentation)
    else:
        filter_labels = segmentation

    ids, sizes = np.unique(filter_labels, return_counts=True)
    filter_ids = ids[sizes < min_size]
    mask = np.in1d(filter_labels, filter_ids).reshape(filter_labels.shape)
    segmentation[mask] = 0

    return measure.label(segmentation)


def extract_data(zip_url, data_dir, project_name):
    """
    Extracts data from `zip_url` to the location identified by `data_dir` parameters.

    Parameters
    ----------
    zip_url: string
        Indicates the external url from where the data is downloaded
    data_dir: string
        Indicates the path to the directory where the data should be saved.
    Returns
    -------

    """
    if not os.path.exists(os.path.join(data_dir, project_name)):
        if os.path.isdir(data_dir):
            pass
        else:
            os.makedirs(data_dir)
            print(f"Created new directory {data_dir}")

        with urlopen(zip_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(data_dir)
        print(f"Downloaded and unzipped data to the location {data_dir}")
    else:
        print(
            "Directory already exists at the location "
            f"{os.path.join(data_dir, project_name)}"
        )


def visualize_2d(
    image,
    top_right,
    bottom_left,
    bottom_right,
    top_right_label,
    bottom_left_label,
    bottom_right_label,
    image_cmap="magma",
    top_right_cmap=None,
    bottom_left_cmap=None,
    bottom_right_cmap=None,
):
    """
    Visualizes 2 x 2 grid with Top-Left (Image)

    Parameters
    -------

    image: Numpy Array (YX)
        Raw Image
    embedding: Numpy Array (3 H W)
        Predicted Embedding
    Returns
    -------

    """

    font = {
        "family": "serif",
        "color": "white",
        "weight": "bold",
        "size": 16,
    }
    plt.figure(figsize=(15, 15))
    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(221)
    plt.imshow(img_show, cmap=image_cmap)
    plt.text(30, 30, "IM", fontdict=font)
    plt.axis("off")
    plt.subplot(222)
    plt.axis("off")
    plt.imshow(top_right, interpolation="None", cmap=top_right_cmap)
    plt.text(30, 30, top_right_label, fontdict=font)
    plt.subplot(223)
    plt.axis("off")
    plt.imshow(bottom_left, interpolation="None", cmap=bottom_left_cmap)
    plt.text(30, 30, bottom_left_label, fontdict=font)
    plt.subplot(224)
    plt.axis("off")
    plt.imshow(bottom_right, interpolation="None", cmap=bottom_right_cmap)
    plt.text(30, 30, bottom_right_label, fontdict=font)
    plt.tight_layout()
    plt.show()
