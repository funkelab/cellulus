from colocseg.utils import get_augmentation_transform, smooth_boundary_fn, sizefilter, CropAndSkipIgnore
from colocseg.transforms import AffinityTf, StardistTf, ThreeclassTf, CellposeTf
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt
from imgaug import augmenters as iaa
import zarr


class CoordinateDataset(Dataset):

    def __init__(self,
                 dataset,
                 output_shape,
                 positive_radius,
                 density=0.2,
                 return_segmentation=True):
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
        self.unbiased_shape = tuple(int(_ - (2 * positive_radius)) for _ in output_shape)
        self.positive_radius = int(positive_radius)

    def sample_offsets_within_radius(self, radius, n_offsets):

        # sample 2 times more samples in a square
        # so that we have room to filter out the ones outside the circle
        offs_x = np.random.randint(-radius, radius + 1,
                                   size=2 * n_offsets)
        offs_y = np.random.randint(-radius, radius + 1,
                                   size=2 * n_offsets)

        offs_coord = np.stack((offs_x, offs_y), axis=1)
        in_circle = (offs_coord**2).sum(axis=1) < radius ** 2
        offs_coord = offs_coord[in_circle]
        not_zero = np.absolute(offs_coord).sum(axis=1) > 0
        offs_coord = offs_coord[not_zero]

        if len(offs_coord) < n_offsets:
            # if the filter removed more than half of all samples
            # repeat
            return self.sample_offsets_within_radius(radius, n_offsets)

        return offs_coord[:n_offsets]

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

        anchor_coord_x = np.random.randint(self.positive_radius,
                                           self.output_shape[0] - self.positive_radius + 1,
                                           size=n_anchor)
        anchor_coord_y = np.random.randint(self.positive_radius,
                                           self.output_shape[1] - self.positive_radius + 1,
                                           size=n_anchor)
        anchor_coord = np.stack((anchor_coord_x, anchor_coord_y), axis=1)

        anchor_samples = np.repeat(anchor_coord, n_reference, axis=0)

        offset_in_pos_radius = self.sample_offsets_within_radius(self.positive_radius,
                                                                 len(anchor_samples))
        refernce_samples = anchor_samples + offset_in_pos_radius

        return anchor_samples, refernce_samples

    def get_num_anchors(self):
        return int(self.density * self.unbiased_shape[0] * self.unbiased_shape[1])

    def get_num_references(self):
        return int(self.density * self.positive_radius**2 * np.pi)

    def get_num_samples(self):
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
        return len(self.root_dataset)

    def __getitem__(self, index):

        x, y = self.unpack(self.root_dataset[index])

        anchor_coordinates, refernce_coordinates = self.sample_coordinates()
        if self.return_segmentation:
            return x, anchor_coordinates, refernce_coordinates, y
        else:
            return x, anchor_coordinates, refernce_coordinates


class TissueNetDataset(Dataset):

    def __init__(self,
                 data_file,
                 tissue_type=None,
                 target_type="cell",
                 target_transform=None,
                 crop_to=None,
                 augment=True,
                 smooth_boundaries=False,
                 min_size=25):

        super().__init__()

        self.target_transform = target_transform
        if target_transform == 'threeclass':
            self.target_tf = ThreeclassTf(inner_distance=2)
        elif target_transform == 'affinity':
            self.target_tf = AffinityTf()
        elif target_transform == 'stardist':
            self.target_tf = StardistTf()
        elif target_transform == 'cellpose':
            self.target_tf = CellposeTf()

        self.tissue_type = tissue_type
        self.target_type = target_type
        self.data_file = data_file
        self.load_data()
        self.augment = augment
        self.apply_smooth_boundaries = smooth_boundaries
        self.min_size = min_size
        self.valid_crop = 46

        if crop_to is not None:
            self.crop_fn = iaa.CropToFixedSize(width=crop_to[0], height=crop_to[1])
        else:
            self.crop_fn = None

        self.batch_augmentation_fn = get_augmentation_transform()

    def __len__(self):
        if not hasattr(self, "_length"):
            self._length = self.raw_data.shape[0]
        return self._length

    def load_data(self):
        with np.load(self.data_file) as data:
            if self.tissue_type is None:
                self.raw_data = data['X']
                self.gt_data = data['y']
            else:
                mask = data["tissue_list"] == self.tissue_type
                self.raw_data = data['X'][mask]
                self.gt_data = data['y'][mask]

    def augment_batch(self, raw, gtseg, idx):
        # prepare images for iaa transform

        raw, gtseg = self.pre_process(raw, gtseg, idx)
        gtseg = gtseg[None]  # CHW -> HWC
        
        if self.augment:
            # make sure that segmentation does not contain negative values
            min_id = gtseg.min()
            gtseg = gtseg - min_id
            raw, gtseg = self.batch_augmentation_fn(image=raw,
                                                    segmentation_maps=gtseg)
            gtseg = gtseg + min_id
            
        if self.crop_fn is not None:
            # make sure that segmentation does not contain negative values
            min_id = gtseg.min()
            gtseg = gtseg - min_id
            raw, gtseg = self.crop_fn(image=raw,
                                      segmentation_maps=gtseg)
            gtseg = gtseg + min_id
            
        raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW

        if gtseg.shape[-1] == 1:
            gtseg = gtseg[0, ..., 0]
        else:
            gtseg = np.transpose(gtseg[0], [2, 0, 1])

        return raw, gtseg

    def smooth_boundaries(self, segmentation):
        segmentation = sizefilter(segmentation, self.min_size)
        segmentation = smooth_boundary_fn(segmentation)
        segmentation = sizefilter(segmentation, self.min_size)
        return segmentation

    def get_gt(self, idx):
        if self.target_type == "cell":
            return self.gt_data[idx, ..., :1].astype(np.int32)
        elif self.target_type == "nucleus":
            return self.gt_data[idx, ..., 1:2].astype(np.int32)
        else:
            raise NotImplementedError(f"unknown target_type: {self.target_type}")

    def pre_process(self, raw, gt, idx):
        if self.valid_crop == 0:
            return raw, gt
        p2d = ((self.valid_crop * 2, self.valid_crop * 2), (self.valid_crop * 2, self.valid_crop * 2), (0, 0))
        raw_padded = np.pad(raw, p2d, mode='constant')
        gt_padded = np.pad(gt, p2d, mode='constant', constant_values=-1)
        return raw_padded, gt_padded
    
    def __getitem__(self, idx):

        raw = self.raw_data[idx]
        gt_segmentation = self.get_gt(idx)
        if self.apply_smooth_boundaries:
            gt_segmentation = self.smooth_boundaries(gt_segmentation)
        raw, gt_segmentation = self.augment_batch(raw, gt_segmentation, idx)

        if self.target_transform is None:
            return raw, gt_segmentation
        else:
            tc = self.target_tf(gt_segmentation)
            return raw, tc, gt_segmentation


class EvenlyMixedDataset(Dataset):
    def __init__(self, datasets, batch_sizes, ):
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        assert len(self.datasets) == len(self.batch_sizes)

    def __getitem__(self, idx):
        batch = []

        for ds, bs in zip(self.datasets, self.batch_sizes):
            samples = []
            for _ in range(bs):
                idx = random.randrange(0, len(ds))
                samples.append(ds[idx])

            batch.append(tuple(np.stack((d[i] for d in samples), axis=0) for i in range(len(samples[0]))))

        return batch

    def __len__(self):
        return max([len(_) for _ in self.datasets]) // sum(self.batch_sizes)


class ZarrSegmentationDataset(TissueNetDataset):

    def __init__(self,
                 data_file,
                 keys,
                 tissue_type=None,
                 target_type="cell",
                 target_transform=None,
                 corrected_instances=None,
                 limit_to_correction=False,
                 crop_to=None,
                 augment=True,
                 smooth_boundaries=False):

        self.keys = keys
        self.tissue_type = tissue_type

        self.corrected_instances = corrected_instances
        self.limit_to_correction = limit_to_correction and corrected_instances is not None
        
        super().__init__(data_file,
                         tissue_type=tissue_type,
                         target_type=target_type,
                         target_transform=target_transform,
                         crop_to=crop_to,
                         augment=augment,
                         smooth_boundaries=smooth_boundaries)
        
        if limit_to_correction:
            self.crop_fn = CropAndSkipIgnore(self.crop_fn)

    
    def pre_process(self, raw, gt, idx):
        
        if self.corrected_instances is None:
            return raw, gt
        
        if self.limit_to_correction:
            gt[:] = -1

        if idx in self.corrected_instances:
            correct_segmentation = self.correction_data[idx]
            
            if self.target_type == "cell":
                correct_segmentation = correct_segmentation[..., :1].astype(np.int32)
            elif self.target_type == "nucleus":
                correct_segmentation = correct_segmentation[..., 1:2].astype(np.int32)
            else:
                raise NotImplementedError(f"unknown target_type: {self.target_type}")

            correction_mask = np.in1d(correct_segmentation.ravel(), self.corrected_instances[idx]).reshape(correct_segmentation.shape)
            
            if not self.limit_to_correction:
                touching_instances = np.unique(gt[correction_mask])
                touching_instances = touching_instances[touching_instances > 0]
                ignore_mask = np.in1d(gt.ravel(), touching_instances).reshape(gt.shape)
                gt[ignore_mask] = -1
            # include gt background close to the corrected instances
            close_to_correction = distance_transform_edt(correction_mask == 0) < 30
            close_background = np.logical_and(correct_segmentation == 0, close_to_correction)
            gt[close_background] = 0
            gt[correction_mask] = correct_segmentation[correction_mask] + gt.max() + 1

        raw, gt = super().pre_process(raw, gt, idx)

        return raw, gt
        
    def load_data(self):
        zin = zarr.open(self.data_file, "r")
    
        selection = None
        if self.tissue_type is not None:
            mask = zin[self.keys["tissue_list"]][:] == self.tissue_type
            selection = np.where(mask)[0]

        if self.limit_to_correction:
            if selection is None:
                print("selecting")
                selection = np.array(list(self.corrected_instances.keys()))
            else:
                print("intersecting")
                to_correct = np.array(list(self.corrected_instances.keys()))
                selection = np.intersect1d(selection, to_correct)

            remapped_instances = {}
            for i, s in enumerate(selection):
                remapped_instances[i] = self.corrected_instances[s]
                
            self.corrected_instances = remapped_instances
        
        if selection is None:
            self.raw_data = zin[self.keys["raw"]]
            self.gt_data = zin[self.keys["gt"]]
            if "correction" in self.keys:
                self.correction_data = zin[self.keys["correction"]]

        else:
            selection_mask = np.in1d(np.arange(len(zin[self.keys["raw"]])), selection)
            self.raw_data = zin[self.keys["raw"]].get_orthogonal_selection(selection_mask)
            self.gt_data = zin[self.keys["gt"]].get_orthogonal_selection(selection_mask)
            if "correction" in self.keys:
                self.correction_data = zin[self.keys["correction"]].get_orthogonal_selection(selection_mask)
            
class ZarrRegressionDataset(Dataset):

    def __init__(self,
                 data_file,
                 keys,
                 tissue_type=None,
                 target_type="cell",
                 crop_to=None,
                 augment=True):

        super().__init__()

        self.keys = keys
        self.tissue_type = tissue_type
        self.target_type = target_type
        self.data_file = data_file
        self.load_data()
        self.augment = augment

        if crop_to is not None:
            self.crop_fn = iaa.CropToFixedSize(width=crop_to[0], height=crop_to[1])
        else:
            self.crop_fn = None

        self.batch_augmentation_fn = get_augmentation_transform(simple=True)

    def load_data(self):
        zin = zarr.open(self.data_file, "r")
        if self.tissue_type is None:
            self.raw_data = zin[self.keys["raw"]]
            self.target_data = zin[self.keys["target"]]
        else:
            sel = np.where(zin[self.keys["tissue_list"]][:] == self.tissue_type)[0]
            self.raw_data = zin[self.keys["raw"]].get_orthogonal_selection(sel)
            self.target_data = zin[self.keys["target"]].get_orthogonal_selection(sel)

    def get_target(self, idx):
        # expect TissueNet Dataset format N, W, H, C
        return np.transpose(self.target_data[idx], [1, 2, 0])

    def __len__(self):
        if not hasattr(self, "_length"):
            self._length = self.raw_data.shape[0]
        return self._length

    def augment_batch(self, raw, target):
        # prepare images for iaa transform
        split_c = raw.shape[-1]
        if target.ndim == 2:
            target = target[..., None]
        raw_target = np.concatenate([raw, target], axis=-1)
        if self.augment:
            raw_target = self.batch_augmentation_fn(image=raw_target)
        if self.crop_fn is not None:
            raw_target = self.crop_fn(image=raw_target)
        raw = raw_target[..., :split_c]
        target = raw_target[..., split_c:]
        raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
        target = np.transpose(target, [2, 0, 1])  # HWC -> CHW
        return raw, target

    def __getitem__(self, idx):

        raw = self.raw_data[idx]
        tc = self.get_target(idx)
        raw, tc = self.augment_batch(raw, tc)

        return raw, tc, None


class ZarrImageDataset(Dataset):

    def __init__(self,
                 data_file,
                 key,
                 crop_to=None,
                 augment=True):

        super().__init__()

        self.data_file = data_file
        self.key = key
        self.load_data()
        self.augment = augment
        if crop_to is not None:
            self.crop_fn = iaa.CropToFixedSize(width=crop_to[0], height=crop_to[1])
        else:
            self.crop_fn = None

        self.batch_augmentation_fn = get_augmentation_transform(medium=True)

    def load_data(self):
        zin = zarr.open(self.data_file, "r")
        img_keys = [f"{self.key}/{k}" for k in zin[self.key] if k.startswith("raw")]
        print("reading ", img_keys)
        self.raw_data = [zin[rk] for rk in img_keys]

    def __len__(self):
        if not hasattr(self, "_length"):
            self._length = sum([len(_) for _ in self.raw_data])
        return self._length

    def augment_batch(self, raw):
        # prepare images for iaa transform
        raw = np.transpose(raw, [1, 2, 0])  # HWC -> CHW
        if self.augment:
            raw = self.batch_augmentation_fn(image=raw)
        if self.crop_fn is not None:
            raw = self.crop_fn(image=raw)
        raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
        return raw.copy()

    def __getitem__(self, idx):
        f=0
        while idx >= len(self.raw_data[f]):
            idx = idx - len(self.raw_data[f])
            f += 1
        
        raw = self.raw_data[f][idx]
        raw = self.augment_batch(raw)
        
        return raw, np.zeros(raw.shape[1:]).astype(np.int32)
