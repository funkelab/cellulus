from torch.utils.data.dataset import Dataset
import numpy as np
import torch

class SimplifiedCoordinateDataset(Dataset):

    def __init__(self,
                 dataset,
                #  output_shape=(100,100,100),
                 output_shape=(128,128,128),
                 positive_radius = (10,10,10),
                 density=0.2,
                 return_segmentation=False):
        """
        max_imbalance_dist: int
            a patch on the edge of the image has an imabalanced
            set of neighbours. We compute the distance to the averge
            neigbour (zero for patches in the center)
            If the average vector is longer than
            max_imbalance_dist the connection is removed
        """

        self.root_dataset = dataset
        self.return_segmentation = return_segmentation
        self.density = density
        self.output_shape = tuple(int(_) for _ in output_shape)
        # self.output_shape = output_shape
        self.positive_radius = tuple(int(_) for _ in positive_radius)
        # self.positive_radius = positive_radius
        self.unbiased_shape = tuple(int(os - (2 * pr)) for os, pr in zip(self.output_shape, self.positive_radius))
        # self.unbiased_shape = int(self.output_shape - (2 * self.positive_radius))

        self.length = 50

    def sample_offsets(self, n_offsets):

        if len(self.output_shape) == 2:
            alpha = np.random.random(size=n_offsets) * np.pi * 2
            r = 1 + np.random.random(size=n_offsets) * (self.positive_radius - 2)
            offs_coord = np.stack((r * np.sin(alpha), r * np.cos(alpha)), axis=1)
            offs_coord = np.round(offs_coord).astype(np.int32)
        elif len(self.output_shape) == 3:
            phi = np.random.random(size=n_offsets) * np.pi * 2
            theta = np.arccos((np.random.random(size=n_offsets) * 2) - 1)
            u = np.random.random(size=n_offsets)
            r = [1 + (pr - 1) * np.cbrt(u) for pr in self.positive_radius]
            
            offs_coord = np.stack((r[0] * np.cos(theta),
                                   r[1] * np.sin(theta) * np.sin(phi),
                                   r[2] * np.sin(theta) * np.cos(phi)), axis=1)
            offs_coord = np.round(offs_coord).astype(np.int32)
        else:
            raise NotImplementedError(f"sample_offsets_within_radius not implemented for d={len(self.output_shape)}")


        return offs_coord

    def sample_coordinates(self):
        """samples pairs of corrdinates within outputshape.
        Returns:
            anchor_samples: uniformly sampled coordinates where all points within
                              the positive radius are still within the outputshape
                              anchor_samples.shape = (p, 2)
            reference_samples: random coordinates in full outputshape space.
                               all_samples.shape = (p, 2)
        """

        n_anchor = self.get_num_anchors()
        n_reference = self.get_num_references()

        # print('in sample_coordinates, n_references =',n_reference,'and n_anchor =',n_anchor)

        anchor_coord = np.stack((np.random.randint(self.positive_radius[i],
                                                   self.output_shape[i] - self.positive_radius[i],
                                                   size=n_anchor) for i in range(len(self.output_shape))), axis=1)

        anchor_samples = np.repeat(anchor_coord, n_reference, axis=0)
        offset_in_pos_radius = self.sample_offsets(len(anchor_samples))
        refernce_samples = anchor_samples + offset_in_pos_radius

        # make sure that we are inbound
        for i in range(len(self.output_shape)):
            mask = refernce_samples[:, i] >= self.output_shape[i]
            refernce_samples[mask, i] = self.output_shape[i] - 1
            mask = refernce_samples[:, i] < 0
            refernce_samples[mask, i] = 0
        
        return anchor_samples, refernce_samples

    def get_num_anchors(self):
        return 16
        return int(self.density * np.product(self.unbiased_shape))

    def get_num_references(self):
        return 16
        if len(self.positive_radius) == 2 or self.positive_radius[0] == 0:
            return int(self.density * np.product(self.positive_radius[-2:]) * np.pi)
        elif len(self.positive_radius) == 3:
            return int(self.density * np.product(self.positive_radius) * np.pi * 4 / 3)
        else:
            raise NotImplementedError("unknown dimension for radius specified")

    def get_num_samples(self):
        # print('get_num_samples requested')
        return self.get_num_anchors() * self.get_num_references()

    def unpack(self, sample):
        
        if isinstance(sample, tuple):
            if len(sample) == 2:
                x, y = sample
            else:
                x = sample[0]
                y = 0.
        else:
            x = sample
            y = 0.

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return x, y

    def __len__(self):
        # print('len requested')
        return len(self.root_dataset)

    def __getitem__(self, index):
        # print('getting item')
        if index>self.length:
            return int([])
        x, y = self.unpack(self.root_dataset[index])
        anchor_coordinates, refernce_coordinates = self.sample_coordinates()
        # print('item got')
        if self.return_segmentation:
            return x, anchor_coordinates, refernce_coordinates, y
        else:
            return x, anchor_coordinates, refernce_coordinates