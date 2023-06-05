import collections
import numpy as np
import torch
from torchvision.transforms import transforms as T

class RandomRotationsAndFlips(T.RandomRotation):
    """
    A class used to represent Random Rotations and Flips for Augmenting 2D Image Data
    ...
    Attributes
    ----------
    keys : dictionary
        See `TwoDimensionalDataset.py`
    Methods
    -------
    __call__: Returns rotated or flipped image
    """

    def __init__(self, keys=[], *args, **kwargs):
        """
        Parameters
        ----------
        keys : dictionary
            keys include `
            See `TwoDimensionalDataset.py`


        """

        super().__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, sample):
        """
            Parameters
            ----------
            sample

            Returns
            ----------
            sample

        """

        times = np.random.choice(4)
        flip = np.random.choice(2)

        for idx, k in enumerate(self.keys):
            assert (k in sample)
            temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
            if flip == 0:
                sample[k] = temp
            else:
                sample[k] = np.ascontiguousarray(np.flip(temp, axis=1))  # flip about Y - axis
        return sample


class RandomRotationsAndFlips_3d(T.RandomRotation):
    """
        A class used to represent Random Rotations and Flips for Augmenting 3D Image Data

        ...

        Attributes
        ----------
        keys : dictionary
            See `ThreeDimensionalDataset.py`

        Methods
        -------
        __call__: Returns rotated or flipped image
        """

    def __init__(self, keys=[], *args, **kwargs):
        """
        Parameters
        ----------
        keys : dictionary
            See `ThreeDimensionalDataset.py`

        """

        super().__init__(*args, **kwargs)
        self.keys = keys


    def __call__(self, sample):
        """
        Parameters
        ----------
        sample

        Returns
        ----------
        sample

        """
        times = np.random.choice(4)
        flip = np.random.choice(2)
        dir_rot = np.random.choice(3)
        dir_flip = np.random.choice(3)

        for idx, k in enumerate(self.keys):

            assert (k in sample)
            if dir_rot == 0:  # rotate about ZY
                temp = np.ascontiguousarray(np.rot90(sample[k], 2 * times, (1, 2)))
            elif dir_rot == 1:  # rotate about YX
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (2, 3)))
            elif dir_rot == 2:  # rotate about ZX
                temp = np.ascontiguousarray(np.rot90(sample[k], 2 * times, (3, 1)))

            if flip == 0:
                sample[k] = temp
            else:
                if dir_flip == 0:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=1))  # Z
                elif dir_flip == 1:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=2))  # Y
                elif dir_flip == 2:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=3))  # X

        return sample


class ToTensorFromNumpy(object):
    """
        A class used to convert numpy arrays to PyTorch tensors

        ...

        Attributes
        ----------
        keys : dictionary
            keys include `instance`, `label`, `center-image`, `image`
        type : str

        normalization_factor: float

        Methods
        -------
        __call__: Returns Pytorch Tensors

            """

    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert (len(keys) == len(type))

        self.keys = keys
        self.type = type


    def __call__(self, sample):

        for idx, k in enumerate(self.keys):
            # assert (k in sample)

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]
            if (k in sample):
                if k == 'image':  # image
                    sample[k] = torch.from_numpy(sample[k].astype("float32")).float()

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)