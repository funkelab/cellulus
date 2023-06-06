import numpy as np
import os
import random
import zarr
from glob import glob
from torch.utils.data import Dataset
from cellulus.utils.utils import normalize_min_max_percentile


class Dataset2D(Dataset):
    """
    A class used to create a PyTorch Dataset for handling 2D images.

    Attributes:
    ----------¬
    image_list: list
        list of strings containing paths to the images

    real_size: int
        Actual number of images (crops) available.

    transform: PyTorch transform

    norm: str
        Set to `min-max-percentile` by default.
        In principle, other normalization strategies could be present.

    Methods
    ----------
    __init__: Initializes an object of class `TwoDimensionalDataset`

    __len__: Returns `self.real_size`

    __getitem__: Returns `sample` (dictionary) which has `image` and `im_name` as keys
    """

    def __init__(
        self,
        data_dir,
        type="train",
        norm="min-max-percentile",
        crop_size=252,
        transform=None,
    ):
        print(
            "2D data loader created! Accessing data from {}/".format(
                data_dir + "/" + type + ".zarr"
            )
        )

        # get image list
        self.raw = zarr.open(os.path.join(data_dir, type + ".zarr"))["raw"]
        print(
            "Number of images in the `{}` directory is {}".format(
                data_dir + "/" + type + ".zarr", self.raw.shape[0]
            )
        )
        self.type = type
        self.real_size = self.raw.shape[0]
        self.transform = transform
        self.norm = norm
        self.crop_size = crop_size

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        index = random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        image = self.raw[index]  # YXC
        if self.norm == "min-max-percentile":
            image = normalize_min_max_percentile(
                image, 1, 99.8, axis=(0, 1)
            )  # TODO (is this correctly done?)
        # add an additional channel
        image = np.transpose(image, (2, 0, 1))  # CYX , where C = 2
        sample["image"] = image[
            :, : self.crop_size, : self.crop_size
        ]  # TODO (cropping is not randomly done at the moment)

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample
