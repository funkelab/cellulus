import numpy as np
import os
import random
import tifffile
from cellulus.utils.utils import normalize_min_max_percentile
from glob import glob


class TwoDimensionalDataset(Dataset):
    """
    A class used to create a PyTorch Dataset for handling 2D images.

    Attributes:
    ----------¬
    image_list: list
        list of strings containing paths to the images

    size: int
        This is set equal to a different value than `real_size`, in case the epoch is designed to not be equal
        to all the data available
    real_size: int
        Actual number of images (crops) available.

    transform: PyTorch transform

    norm: str
        Set to `min-max-percentile` by default.
        In principle, other normalization strategies could be present.

    Methods
    ----------
    __init__: Initializes an object of class `TwoDimensionalDataset`

    __len__: Returns `self.real_size` if `self.size` is None

    __getitem__: Returns `sample` (dictionary) which has `image` and `im_name` as keys
    """

    def __init__(self, data_dir, transform=None, norm='min-max-percentile'):
        print('2D data loader created! Accessing data from {}/'.format(data_dir))

        # get image list
        image_list = sorted(glob(os.path.join(data_dir, 'images', '*.tif')))
        print('Number of images in the `{}` directory is {}'.format(data_dir, len(image_list)))
        self.image_list = image_list

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.norm = norm
        self.crop_size = crop_size

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image (currently configured for `tiff` images)

        image = tifffile.imread(self.image_list[index])  # YX

        # add an additional channel
        image = image[np.newaxis, ...]  # CYX , where C = 1
        if self.norm == 'min-max-percentile':
            image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample
