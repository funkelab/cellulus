import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import zarr
from skimage.io import imsave
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from colocseg.loss import AnchorLoss
from colocseg.loss_supervised import SupervisedInstanceEmbeddingLoss
from colocseg.model import Unet2D
from colocseg.utils import BuildFromArgparse, import_by_string, zarr_append


class SSLTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self,
                 in_channels=1,
                 out_channels=18,
                 initial_lr=1e-4,
                 regularization=1e-4,
                 temperature=10,
                 coordinate_offset_after_valid_unet=8,
                 lr_milestones=(100,)
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_lr = initial_lr
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.regularization = regularization
        self.temperature = temperature
        self.coordinate_offset_after_valid_unet = coordinate_offset_after_valid_unet
        self.build_models()
        self.build_loss()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--out_channels', type=int)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--regularization', type=float, default=1e-4)
        parser.add_argument('--lr_milestones', nargs='*', default=[10000, 20000])
        parser.add_argument('--temperature', type=float, default=10)
        parser.add_argument('--coordinate_offset_after_valid_unet', type=int, default=8)
        return parser

    def forward(self, x):
        return self.mini_unet(x)

    def build_models(self):
        self.mini_unet = Unet2D(
            self.in_channels,
            self.out_channels,
            num_fmaps=256)

    def build_loss(self, ):
        self.validation_loss = SupervisedInstanceEmbeddingLoss(30.)
        self.anchor_loss = AnchorLoss(self.temperature)

    def training_step(self, batch, batch_nb):

        x, anchor_coordinates, refernce_coordinates = batch
        emb_relative = self.mini_unet.forward(x)
        emb_anchor = self.mini_unet.select_and_add_coordinates(emb_relative, anchor_coordinates)
        emb_ref = self.mini_unet.select_and_add_coordinates(emb_relative, refernce_coordinates)
        anchor_loss = self.anchor_loss(emb_anchor, emb_ref)

        self.log_images(x, emb_relative)

        self.log(
            'anchor_loss',
            anchor_loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        self.log(
            'mean_offset',
            emb_relative.detach().mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True)
        self.log(
            'max_offset',
            emb_relative.detach().abs().max(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True)
        self.log(
            'anchor_loss_temperature',
            self.anchor_loss.temperature,
            on_step=True,
            prog_bar=True,
            logger=True)

        if self.regularization > 0.:
            reg_loss = self.regularization * emb_anchor.norm(2, dim=-1).sum()
            loss = anchor_loss + reg_loss
            self.log('reg_loss', reg_loss.detach(), on_step=True, prog_bar=True, logger=True)
        else:
            loss = anchor_loss

        tensorboard_logs = {'train_loss': loss.detach()}
        tensorboard_logs['iteration'] = self.global_step
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        x, anchor_coordinates, refernce_coordinates, y = batch

        with torch.no_grad():

            # pad image to get full output
            anchor_coordinates = anchor_coordinates + self.coordinate_offset_after_valid_unet
            coavu = self.coordinate_offset_after_valid_unet
            p2d = (coavu, coavu, coavu, coavu)
            x_padded = F.pad(x, p2d, mode='reflect')

            embedding = self.mini_unet.forward(x_padded)
            emb_anchor = self.mini_unet.select_and_add_coordinates(embedding, anchor_coordinates)
            emb_ref = self.mini_unet.select_and_add_coordinates(embedding, refernce_coordinates)
            loss = self.anchor_loss(emb_anchor, emb_ref)

            self.log('val_anchor_loss', loss.detach(), on_epoch=True, prog_bar=False, logger=True)
            for margin in [1., 5., 10, 20., 40]:
                self.validation_loss.push_margin = margin
                absoute_embedding = self.anchor_loss.absoute_embedding(emb_anchor, emb_ref)
                pull_loss, push_loss = self.validation_loss(emb_anchor, anchor_coordinates, y, split_pull_push=True)

                self.log(
                    f'val_clustering_loss_margin_pull_{margin}',
                    pull_loss.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    logger=True)
                self.log(
                    f'val_clustering_loss_margin_push_{margin}',
                    push_loss.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    logger=True)
                self.log(
                    f'val_clustering_loss_margin_both_{margin}',
                    (pull_loss + push_loss).detach(),
                    on_epoch=True,
                    prog_bar=False,
                    logger=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        # scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        return optimizer#], [scheduler]

    def log_images(self, raw, network_prediction):
        from pathlib import Path
        import numpy as np
        import math
        if self.global_step > 0 and (math.log2(self.global_step).is_integer() or self.global_step % 10000 == 0):
            Path("img").mkdir(exist_ok=True)
            x = raw.detach().cpu().numpy()
            x = np.reshape(np.transpose(x, (0, 2, 1, 3)), (-1, x.shape[1] * x.shape[3]))
            imsave(f"img/raw_ssl_{self.global_step}.png", x, check_contrast=False)

            x = network_prediction.detach().cpu().numpy()
            x = np.reshape(np.transpose(x, (0, 2, 1, 3)), (-1, x.shape[1] * x.shape[3]))
            imsave(f"img/pred_ssl_{self.global_step}.png", x, check_contrast=False)

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


class PartiallySupervisedTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self,
                 in_channels=1,
                 out_channels=17,
                 loss_name_super="StardistLoss",
                 loss_name_aux="StardistLoss",
                 aux_channels=17,
                 unet_head_type="single",
                 unet_fmap_inc_factor=3,
                 loss_alpha=0.01,
                 loss_delay=0,
                 loss_same_channel=False,
                 valid_crop=46,
                 initial_lr=1e-4,
                 lr_milestones=(100,),
                 train_without_pseudo_gt=False,
                 save_batches=True
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_name_super = loss_name_super
        self.loss_name_aux = loss_name_aux
        self.aux_channels = aux_channels
        self.loss_alpha = loss_alpha
        self.loss_delay = loss_delay
        self.loss_same_channel = loss_same_channel
        self.unet_head_type = unet_head_type
        self.unet_fmap_inc_factor = unet_fmap_inc_factor
        self.valid_crop = valid_crop
        self.initial_lr = initial_lr
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.metrics = {}
        self.train_without_pseudo_gt = train_without_pseudo_gt
        self.save_batches = save_batches

        self.build_models()
        self.build_loss()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--out_channels', type=int)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--valid_crop', type=int, default=46)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--loss_name_super', type=str, default="StardistLoss")
        parser.add_argument('--loss_name_aux', type=str, default="StardistLoss")
        parser.add_argument('--unet_head_type', type=str, default="single")
        parser.add_argument('--aux_channels', type=int, default=17)
        parser.add_argument('--loss_alpha', type=float, default=0.01)
        parser.add_argument('--loss_delay', type=int, default=0)
        parser.add_argument('--unet_fmap_inc_factor', type=int, default=3)
        parser.add_argument('--loss_same_channel', action='store_true')

        parser.add_argument('--lr_milestones', nargs='*', default=[10000, 20000])
        parser.add_argument('--train_without_pseudo_gt', action='store_true')

        return parser

    def forward(self, x):
        return self.maxi_unet(x)

    def build_models(self):
        self.maxi_unet = Unet2D(
            self.in_channels,
            self.out_channels,
            self.aux_channels,
            head_type=self.unet_head_type,
            num_fmaps=32,
            fmap_inc_factor=self.unet_fmap_inc_factor,
            depth=3)

    def build_loss(self, ):
        loss_class_super = import_by_string(f'colocseg.loss.{self.loss_name_super}')
        self.criterion_super = loss_class_super()
        loss_class_aux = import_by_string(f'colocseg.loss.{self.loss_name_aux}')
        self.criterion_aux = loss_class_aux()

    def log_metrics(self):
        for k in self.metrics:
            try:
                self.log(k, self.metrics[k], prog_bar=k == "f1")
            except:
                pass
        self.metrics = {}

    def crop_to_valid(self, tensor):
        return tensor[..., self.valid_crop:-self.valid_crop, self.valid_crop:-self.valid_crop]

    def log_batch(self, filename, log_now=False, **kwargs):
        if self.save_batches and self.global_step % 5000 == 0 or log_now:
            zout = zarr.open(filename)
            for k in kwargs:
                if kwargs[k] is not None:
                    zout[k] = kwargs[k].detach().cpu().numpy()

    def log_images(self, name, raw, network_prediction, target, gtseg):
        from pathlib import Path
        import numpy as np
        import math
        if self.global_step > 0 and math.log2(self.global_step).is_integer():
            Path("img").mkdir(exist_ok=True)
            x = raw.detach().cpu().numpy()
            x = np.reshape(np.transpose(x, (0, 2, 1, 3)), (-1, x.shape[1] * x.shape[3]))
            imsave(f"img/raw_{name}_{self.global_step}.png", x, check_contrast=False)

            fgbg0 = network_prediction.detach().cpu()[:, 0].sigmoid().numpy()
            fgbg0 = np.reshape(fgbg0, (-1, fgbg0.shape[-1]))
            pred0 = network_prediction.detach().cpu()[:, 1]
            pred0 = np.reshape(pred0, (-1, pred0.shape[-1]))
            pred0 -= pred0.min()
            pred0 /= pred0.max() + 1e-8
            imsave(f"img/pred_{name}_{self.global_step}.png", np.concatenate(
                    (fgbg0,
                    0 * fgbg0[:1], 0 * fgbg0[:1] + 1, 0 * fgbg0[:1],
                     pred0),
                    axis=0),
                    check_contrast=False)

            g = gtseg.detach().cpu().reshape(-1, gtseg.shape[-1]).numpy()
            # g /= g.max() + 0.01
            imsave(f"img/gt_{name}_{self.global_step}.png", g, check_contrast=False)
            for c in range(target.shape[1]):
                t = target[:, c].detach().cpu().reshape(-1, target.shape[-1]).numpy()
                t /= t.max()
                imsave(f"img/target_{name}_{self.global_step}_{c}.png", t, check_contrast=False)


    def training_step(self, batch, batch_nb):

        add_aux_loss = self.loss_alpha > 0 and not self.train_without_pseudo_gt and self.loss_delay <= self.global_step
        add_super_loss = self.loss_alpha < 1. and (self.loss_delay >= 0 or (
            self.loss_delay < 0 and -self.loss_delay <= self.global_step))

        batch_supervised, batch_aux = batch
        loss = 0.

        if add_aux_loss:
            x_aux, target_aux, aux_segmentation = batch_aux

            network_prediction_aux_on_aux, network_prediction_super_on_aux = self.maxi_unet.forward(x_aux)
            network_prediction_aux = network_prediction_aux_on_aux
            if self.loss_same_channel:
                network_prediction_aux = network_prediction_super_on_aux

            self.log_batch(f"batch_{self.global_step}_aux.zarr",
                        x_aux=x_aux,
                        network_prediction_aux=network_prediction_aux,
                        target_aux=target_aux,
                        aux_segmentation=aux_segmentation[:, None] if torch.is_tensor(aux_segmentation) else None)
            target_aux = self.crop_to_valid(target_aux)

            if self.global_step % 5000 == 0:
                def log_hook_full(grad_input):
                    outzarr = zarr.open(f"grad_{self.global_step}_aux.zarr", "w")
                    zarr_append("grad", grad_input.detach().cpu().numpy()[None], outzarr)
                    zarr_append("x_aux", (x_aux).detach().cpu().numpy()[None], outzarr)
                    zarr_append("target_aux", (target_aux).detach().cpu().numpy()[None], outzarr)
                    zarr_append("network_prediction_aux", network_prediction_aux.detach().cpu().numpy()[None], outzarr)
                    handle.remove()
                handle = network_prediction_aux.register_hook(log_hook_full)

            self.log_images("aux", x_aux, network_prediction_aux, target_aux, aux_segmentation)
            loss_pseudo = self.criterion_aux(network_prediction_aux, target_aux)

            self.log("train_loss_pseudo", loss_pseudo.detach().item(), prog_bar=True)
            loss = loss + (self.loss_alpha * loss_pseudo)         

        if add_super_loss:
            x_super, target_super, gt_segmentation = batch_supervised
            network_prediction_super = self.maxi_unet.forward(x_super)[1]
            self.log_batch(f"batch_{self.global_step}_super.zarr",
                        x_super=x_super,
                        network_prediction_super=network_prediction_super,
                        target_super=target_super,
                        gt_segmentation=gt_segmentation[:, None])
            target_super = self.crop_to_valid(target_super)
            self.log_images("super", x_super, network_prediction_super, target_super, gt_segmentation)

            if self.global_step % 5000 == 0:
                def log_hook_full(grad_input):
                    outzarr = zarr.open(f"grad_{self.global_step}_super.zarr", "w")
                    zarr_append("grad", grad_input.detach().cpu().numpy()[None], outzarr)
                    zarr_append("x_super", (x_super).detach().cpu().numpy()[None], outzarr)
                    zarr_append("target_super", (target_super).detach().cpu().numpy()[None], outzarr)
                    zarr_append("network_prediction_aux",
                                network_prediction_super.detach().cpu().numpy()[None], outzarr)
                    handle2.remove()
                handle2 = network_prediction_super.register_hook(log_hook_full)

            loss_super = self.criterion_super(network_prediction_super, target_super)
            self.log("train_loss_super", loss_super.detach().item(), prog_bar=True)
            loss = loss + ((1 - self.loss_alpha) * loss_super)

        self.log("loss_alpha", self.loss_alpha, prog_bar=False)
        tensorboard_logs = {'train_loss': loss.detach().item()}
        self.log_metrics()
        self.log("train_loss", loss.detach().item())
        
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        return 0.

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=0.0)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        return [optimizer], [scheduler]
