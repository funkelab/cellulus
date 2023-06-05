import collections
import numpy as np
import torch
from torchvision.transforms import transforms as T

class RandomRotationsAndFlips(T.RandomRotation):
    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, sample):
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