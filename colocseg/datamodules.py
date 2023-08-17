import argparse
import pytorch_lightning as pl
import torch
import imgaug as ia
import os
import json
import random
import zarr
from torch.utils.data import DataLoader, ConcatDataset, Subset
from colocseg.datasets import (CoordinateDataset, TissueNetDataset, ZarrSegmentationDataset,
                               EvenlyMixedDataset, ZarrRegressionDataset, ZarrImageDataset)
import numpy as np


class AnchorDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, dspath,
                 shape=(256, 256), output_shape=(240, 240),
                 loader_workers=10, positive_radius=32):

        super().__init__()
        self.batch_size = batch_size
        self.dspath = dspath
        self.shape = tuple(int(_) for _ in shape)
        self.output_shape = tuple(int(_) for _ in output_shape)
        self.loader_workers = loader_workers
        output_shape
        self.positive_radius = positive_radius

    def setup_datasets(self):
        raise NotImplementedError()

    def setup(self, stage=None):

        img_ds_train, img_ds_val = self.setup_datasets()

        self.ds_train = CoordinateDataset(
            img_ds_train,
            self.output_shape,
            self.positive_radius,
            density=0.1,
            return_segmentation=False)

        self.ds_val = CoordinateDataset(
            img_ds_val,
            self.output_shape,
            self.positive_radius,
            density=0.1,
            return_segmentation=True)

        return (img_ds_train, img_ds_val), (self.ds_train, self.ds_val)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          drop_last=False)

    def test_dataloader(self):
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--dspath', type=str)
        parser.add_argument('--shape', nargs='*', default=(256, 256))
        parser.add_argument('--output_shape', nargs='*', default=(256, 256))
        parser.add_argument('--positive_radius', type=int, default=64)

        return parser


class TissueNetDataModule(AnchorDataModule):

    def setup_datasets(self):

        train_ds = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_train.npz"),
                                    crop_to=self.shape,
                                    augment=True)

        val_ds = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_val.npz"),
                                  crop_to=(252, 252),
                                  augment=False)

        return ConcatDataset([train_ds, val_ds]), val_ds


class CTCDataModule(AnchorDataModule):

    def setup_datasets(self):

        train_ds = ZarrImageDataset(self.dspath,
                                    "train",
                                    crop_to=self.shape,
                                    augment=True)
        
        val_ds = ZarrImageDataset(self.dspath,
                                  "train",
                                  crop_to=(252, 252),
                                  augment=False)
        
        return train_ds, train_ds

class PartiallySupervisedDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, tissuenet_root,
                 pseudo_gt_root,
                 raw_key,
                 gt_key,
                 pseudo_gt_key,
                 pseudo_gt_val_key=None,
                 pseudo_gt_score_key=None,
                 target_transform="stardist",
                 target_transform_aux="stardist",
                 corrected_instances_file=None,
                 tissue_type=None,
                 target_type="cell",
                 shape=(256, 256),
                 output_shape=(240, 240),
                 limit=None,
                 datasetseed=42,
                 loader_workers=10):

        super().__init__()
        self.batch_size = batch_size
        self.tissuenet_root = tissuenet_root
        self.pseudo_gt_root = pseudo_gt_root
        self.raw_key = raw_key
        self.gt_key = gt_key
        self.pseudo_gt_key = pseudo_gt_key
        
        if pseudo_gt_val_key is None:
            self.pseudo_gt_val_key = pseudo_gt_key
        else:
            self.pseudo_gt_val_key = pseudo_gt_val_key
        self.pseudo_gt_score_key = pseudo_gt_score_key
            
        self.target_transform = target_transform
        self.target_transform_aux = target_transform_aux
        self.corrected_instances_file = corrected_instances_file
        self.shape = tuple(int(_) for _ in shape)
        self.output_shape = tuple(int(_) for _ in output_shape)
        self.loader_workers = loader_workers
        self.limit = limit
        self.seed = datasetseed
        self.tissue_type = tissue_type
        self.target_type = target_type

    def setup_datasets(self):

        aux_is_segmentation = self.target_transform_aux is not None

        if aux_is_segmentation:
            val_ds = ZarrSegmentationDataset(self.pseudo_gt_root,
                                             {"raw": "val/raw",
                                                     "gt": f"val/{self.pseudo_gt_val_key}",
                                                     "tissue_list": "val/tissue_list"},
                                             tissue_type=self.tissue_type,
                                             target_type="cell",
                                             crop_to=(252, 252),
                                             augment=False,
                                             smooth_boundaries=True)
        else:
            val_ds = ZarrRegressionDataset(self.pseudo_gt_root,
                                            {"raw": "val/raw",
                                            "target": f"val/{self.pseudo_gt_val_key}",
                                            "tissue_list": "val/tissue_list"},
                                            tissue_type=self.tissue_type,
                                            target_type="cell",
                                            crop_to=(252, 252),
                                            augment=False)

        corrected_instances = None
        if self.corrected_instances_file is not None:
            if self.corrected_instances_file == 'all':
                
                train_ds = ZarrSegmentationDataset(self.pseudo_gt_root,
                                    {"raw": f"train/{self.raw_key}",
                                    "gt": f"train/{self.gt_key}",
                                    "correction": f"train/{self.gt_key}",
                                    "tissue_list": "train/tissue_list"},
                                    tissue_type=self.tissue_type,
                                    target_type=self.target_type,
                                    target_transform=self.target_transform_aux,
                                    crop_to=self.shape,
                                    augment=True,
                                    smooth_boundaries=True)
                
                return train_ds, train_ds, val_ds
            else:
                with open(self.corrected_instances_file, 'r') as fp:
                    def toint(x):
                        return {int(k): v for k, v in x}
                    corrected_instances = json.load(fp, object_pairs_hook=toint)

        train_ds = ZarrSegmentationDataset(self.pseudo_gt_root,
                                           {"raw": f"train/{self.raw_key}",
                                            "gt": f"train/{self.gt_key}",
                                            "correction": f"train/{self.gt_key}",
                                            "tissue_list": "train/tissue_list"},
                                           tissue_type=self.tissue_type,
                                           target_type=self.target_type,
                                           target_transform=self.target_transform_aux,
                                           corrected_instances=corrected_instances,
                                           limit_to_correction=True,
                                           crop_to=self.shape,
                                           augment=True,
                                           smooth_boundaries=True)

        if aux_is_segmentation:
            train_pseudo_ds = ZarrSegmentationDataset(self.pseudo_gt_root,
                                                      {"raw": f"train/{self.raw_key}",
                                                       "gt": f"train/{self.pseudo_gt_key}",
                                                       "correction": f"train/{self.gt_key}",
                                                       "tissue_list": "train/tissue_list"},
                                                      tissue_type=self.tissue_type,
                                                      target_type="cell",
                                                      target_transform=self.target_transform_aux,
                                                      corrected_instances=corrected_instances,
                                                      limit_to_correction=False,
                                                      crop_to=self.shape,
                                                      augment=True,
                                                      smooth_boundaries=True)

        else:
            train_pseudo_ds = ZarrRegressionDataset(self.pseudo_gt_root,
                                                    {"raw": f"train/{self.raw_key}",
                                                     "target": f"train/{self.pseudo_gt_key}",
                                                     "tissue_list": "train/tissue_list"},
                                                    tissue_type=self.tissue_type,
                                                    target_type="cell",
                                                    crop_to=self.shape,
                                                    augment=True)

        return train_ds, train_pseudo_ds, val_ds

    def setup(self, stage=None):

        ds_train, ds_pseudo, ds_val = self.setup_datasets()
        self.ds_train = ds_train
        self.ds_pseudo = ds_pseudo
        self.ds_val = ds_val

    def train_dataloader(self):

        if self.limit is None:
            mixed_train_ds = EvenlyMixedDataset([self.ds_train, self.ds_pseudo],
                                                [self.batch_size // 2, self.batch_size // 2])
        else:
            assert(len(self.ds_train) == len(self.ds_pseudo))
            assert(len(self.ds_train) >= self.limit)
            
            if self.pseudo_gt_score_key is None:
                supervised_indices = np.random.RandomState(seed=self.seed).permutation(len(self.ds_train))[:self.limit]
                remaining_indices = np.random.RandomState(seed=self.seed).permutation(len(self.ds_train))[self.limit:]
            else:
                z = zarr.open(self.pseudo_gt_root, "r")
                scores = z[self.pseudo_gt_score_key][:]
                assert(z[f"train/{self.pseudo_gt_key}"].shape[0] == scores.shape[0])
                sorted_indices = np.argsort(scores)
                supervised_indices = sorted_indices[:self.limit]
                remaining_indices = sorted_indices[self.limit:]
            
            supervised_limited_train_ds = Subset(self.ds_train, supervised_indices)
            remainin_pseudo_ds = Subset(self.ds_pseudo, remaining_indices)
            supervised_plus_pseudo = ConcatDataset([supervised_limited_train_ds, remainin_pseudo_ds])
            mixed_train_ds = EvenlyMixedDataset([supervised_limited_train_ds, supervised_plus_pseudo],
                                                [self.batch_size // 2, self.batch_size // 2])
            
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            ia.seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(mixed_train_ds,
                          batch_size=None,
                          worker_init_fn=seed_worker,
                          num_workers=self.loader_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          drop_last=False)

    def test_dataloader(self):
        return None

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--tissuenet_root', type=str)
        parser.add_argument('--target_transform', type=str, default="stardist")
        parser.add_argument('--target_transform_aux', type=str, default=None)
        parser.add_argument('--corrected_instances_file', type=str, default=None)
        parser.add_argument('--pseudo_gt_root', type=str)
        parser.add_argument('--gt_key', type=str, default="gt")
        parser.add_argument('--raw_key', type=str, default="raw")
        parser.add_argument('--pseudo_gt_key', type=str)
        parser.add_argument('--pseudo_gt_val_key', type=str, default=None)
        parser.add_argument('--target_type', type=str, default="cell")
        parser.add_argument('--tissue_type', type=str, default=None)
        parser.add_argument('--shape', nargs='*', default=(256, 256))
        parser.add_argument('--output_shape', nargs='*', default=(256, 256))
        parser.add_argument('--limit', type=int, default=None)
        parser.add_argument('--datasetseed', type=int, default=42)
        parser.add_argument('--pseudo_gt_score_key', type=str, default=None)

        return parser
