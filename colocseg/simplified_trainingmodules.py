import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import zarr
from skimage.io import imsave
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import os
from pathlib import Path

from colocseg.loss import AnchorLoss
from colocseg.loss_supervised import SupervisedInstanceEmbeddingLoss
from colocseg.model import Unet2D, Unet3D
from colocseg.utils import BuildFromArgparse, import_by_string, zarr_append


class SimpleSSLTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self,
                 in_channels=1,
                 out_channels=18,
                 initial_lr=1e-4,
                 regularization=1e-4,
                 temperature=10,
                 coordinate_offset_after_valid_unet=8,
                 lr_milestones=(100,),
                 unet_depth=1,
                 volume=False,
                 unet_checkpoint=None,
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_lr = initial_lr
        self.volume = volume
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.regularization = regularization
        self.temperature = temperature
        self.coordinate_offset_after_valid_unet = coordinate_offset_after_valid_unet
        self.unet_depth = unet_depth
        self.unet_checkpoint = unet_checkpoint

        self.build_models()
        self.load_checkpoint()
        self.build_loss()
        self.last_loss = 0.


    def forward(self, x):
        return self.mini_unet(x)

    def build_models(self):
        if self.volume:
            self.mini_unet = Unet3D(
                self.in_channels,
                self.out_channels,
                depth=self.unet_depth,
                num_fmaps=256)
            return
        
        self.mini_unet = Unet2D(
            self.in_channels,
            self.out_channels,
            num_fmaps=256)

    def load_checkpoint(self):
        if self.unet_checkpoint is not None:
            print(f"loading checkpoint from {self.unet_checkpoint}")
            model_state_dict = torch.load(self.unet_checkpoint)["model_state_dict"]
            self.mini_unet.load_state_dict(model_state_dict)

    def build_loss(self, ):
        self.validation_loss = SupervisedInstanceEmbeddingLoss(30.)
        self.anchor_loss = AnchorLoss(self.temperature)

    def training_step(self, batch, batch_nb):

        x, anchor_coordinates, refernce_coordinates = batch
        emb_relative = self.mini_unet.forward(x)
        emb_anchor = self.mini_unet.select_and_add_coordinates(emb_relative, anchor_coordinates)
        emb_ref = self.mini_unet.select_and_add_coordinates(emb_relative, refernce_coordinates)
        anchor_loss = self.anchor_loss(emb_anchor, emb_ref)

        # self.log_images(x, emb_relative)

        # self.log(
        #     'anchor_loss',
        #     anchor_loss.detach(),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True)
        self.last_loss = anchor_loss.detach()
            
        # mean_offset = emb_relative.detach().mean(dim=(0, 2, 3, 4))
        # for i in range(3):
        #     self.log(
        #         f'mean_offset_{i}',
        #         mean_offset[i],
        #         on_step=True,
        #         on_epoch=False,
        #         prog_bar=True,
        #         logger=True)
        # self.log(
        #     'max_dist',
        #     emb_relative.detach().norm(2, dim=1).max(),
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=True,
        #     logger=True)

        if self.regularization > 0.:
            dist = emb_relative.norm(2, dim=1)
            # min_reg_distance = 300
            # reg_loss = self.regularization * dist[dist > min_reg_distance].mean()
            reg_loss = self.regularization * dist.mean()
            
            loss = anchor_loss + reg_loss
            self.log('reg_loss', reg_loss.detach(), on_step=True, prog_bar=True, logger=True)
        else:
            loss = anchor_loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        # scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        return optimizer#], [scheduler]

    def log_images(self, raw, network_prediction):
        from pathlib import Path
        import numpy as np
        import math
        if self.global_step > 0 and (math.log2(self.global_step).is_integer() or self.global_step % 1000 == 0):
            Path("img").mkdir(exist_ok=True)
            x = raw.detach().cpu().numpy()

            if len(x.shape) == 5:
                x = np.transpose(x, (0, 1, 3, 2, 4))
                x = np.reshape(x, (-1, x.shape[-2]* x.shape[-1]))
                imsave(f"img/raw_ssl_{self.global_step}.png", x, check_contrast=False)
            elif len(x.shape) == 4:
                x = np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1))
                imsave(f"img/raw_ssl_{self.global_step}.png", x, check_contrast=False)

            x = network_prediction.detach().cpu().numpy()
            x = np.clip(x, -100, 100)
            if len(x.shape) == 5:
                x = x - np.mean(x, axis=(2,3,4), keepdims=True)
                x = x / np.max(np.abs(x), axis=(2, 3, 4), keepdims=True)
                x = np.transpose(x, (0, 1, 3, 2, 4))
                x = np.reshape(x, (-1, x.shape[-2] * x.shape[-1]))
                print(x.shape)
                imsave(f"img/pred_ssl_{self.global_step}.png", x, check_contrast=False)
            elif len(x.shape) == 4:
                x = np.reshape(np.transpose(x, (0, 2, 1, 3)),
                               (x.shape[0] * x.shape[1] * x.shape[2], -1))
                imsave(f"img/pred_ssl_{self.global_step}.png", x, check_contrast=False)

            model_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                           os.pardir,
                                                           os.pardir,
                                                           "models"))

            if self.global_step > 100:
                os.makedirs(model_directory, exist_ok=True)
                model_save_path = os.path.join(
                    model_directory, f"mini_unet_{self.global_step:08d}_{self.local_rank:02}.torch")
                torch.save({"model_state_dict": self.mini_unet.state_dict()}, model_save_path)


    def log_now(self, val=False):

        if val:
            if self.last_val_log == self.global_step:
                return False
            else:
                self.last_val_log = self.global_step
                return True

        if self.global_step > 1024:
            return self.global_step % 2048 == 0
        else:
            return self.global_step % 64 == 0

