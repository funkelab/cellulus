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
        norm,
        crop_size,
        density,
        radius,
        type="train",
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
        self.density = density
        self.crop_shape = tuple(int(_) for _ in (crop_size, crop_size))  # (252, 252)
        self.radius = radius  # 32
        self.unbiased_shape = tuple(
            int(_ - (2 * radius)) for _ in self.crop_shape
        )  # (188, 188)

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

        (
            anchor_coordinates,
            reference_coordinates,
        ) = self.get_sample_coordinates()  # (N, 2)
        sample["anchor_coordinates"] = anchor_coordinates
        sample["reference_coordinates"] = reference_coordinates

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def get_num_anchors(self):
        return int(self.density * self.unbiased_shape[0] * self.unbiased_shape[1])

    def get_num_references(self):
        return int(self.density * self.radius**2 * np.pi)

    def get_sample_coordinates(self):
        n_anchor = self.get_num_anchors()
        n_reference = self.get_num_references()

        anchor_coord_x = np.random.randint(
            self.radius, self.crop_shape[0] - self.radius + 1, size=n_anchor
        )
        anchor_coord_y = np.random.randint(
            self.radius, self.crop_shape[1] - self.radius + 1, size=n_anchor
        )
        anchor_coord = np.stack((anchor_coord_x, anchor_coord_y), axis=1)

        anchor_samples = np.repeat(anchor_coord, n_reference, axis=0)
        offset_in_radius = self.sample_offsets_within_radius(len(anchor_samples))
        reference_samples = anchor_samples + offset_in_radius

        return anchor_samples, reference_samples

    def sample_offsets_within_radius(self, n_offsets):
        offsets_x = np.random.randint(-self.radius, self.radius + 1, size=2 * n_offsets)
        offsets_y = np.random.randint(-self.radius, self.radius + 1, size=2 * n_offsets)

        offsets_coordinates = np.stack((offsets_x, offsets_y), axis=1)
        in_circle = (offsets_coordinates**2).sum(axis=1) < self.radius**2
        offsets_coordinates = offsets_coordinates[in_circle]
        not_zero = np.absolute(offsets_coordinates).sum(axis=1) > 0
        offsets_coordinates = offsets_coordinates[not_zero]

        # TODO --> are the following two lines needed?
        # if len(offsets_coordinates) < n_offsets:
        #    return self.sample_offsets_within_radius(n_offsets)

        return offsets_coordinates[:n_offsets]
