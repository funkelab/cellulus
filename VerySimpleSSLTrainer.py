import torch
import pytorch_lightning as pl
from gpt_unets import UNet3DFromGPT, UNetFromGPT
from colocseg.loss import AnchorLoss

class VerySimpleSSLTrainer(pl.LightningModule):
    def __init__(self,
                 in_channels=1,
                 out_channels=3,
                 initial_lr=1e-4,
                 volume = True
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial_lr = initial_lr

        self.build_models()
        self.last_loss = 0.
        self.volume = True
        self.temperature = 10

    # def forward(self, x):
    #     print('SSLTrainer forward')
    #     return self.mini_unet(x)

    def build_models(self):
        self.volume = True
        # print('build_models, self.volume = ',self.volume)
        if self.volume:
            self.mini_unet = UNet3DFromGPT(
                self.in_channels,
                self.out_channels)
            return
        
        self.mini_unet = UNetFromGPT(
            self.in_channels,
            self.out_channels,
            )

    def load_checkpoint(self, unet_checkpoint):
        self.unet_checkpoint = unet_checkpoint
        if self.unet_checkpoint is not None:
            # print(f"loading checkpoint from {self.unet_checkpoint}")
            model_state_dict = torch.load(self.unet_checkpoint)["model_state_dict"]
            self.mini_unet.load_state_dict(model_state_dict)

    # def build_loss(self, ):
    #     print('build loss')
    #     self.validation_loss = SupervisedInstanceEmbeddingLoss(30.)
    #     self.anchor_loss = AnchorLoss(self.temperature)

    def training_step(self, batch, batch_nb):
        # print('training_step, len(batch) = ',len(batch))
        x, anchor_coordinates, refernce_coordinates = batch

        embedding = self.mini_unet(x)

        # anchor_coordinates[:,:,0] is all of the x coordinates in our batch
        if self.volume:
            selection1 = embedding[:,:,anchor_coordinates[:,:,0].long(),anchor_coordinates[:,:,1].long(),anchor_coordinates[:,:,2].long()].squeeze()
        else:
            selection1 = embedding[:,:,anchor_coordinates[:,:,1].long(),anchor_coordinates[:,:,0].long()].squeeze()
        if len(selection1.shape)==1:
            selection1 = selection1.unsqueeze(1)
        
        selection1 = selection1.transpose(1, 0)

        selection1[:,0] += anchor_coordinates[:,:,0].squeeze()
        selection1[:,1] += anchor_coordinates[:,:,1].squeeze()
        selection1[:,2] += anchor_coordinates[:,:,2].squeeze()        
        # !!!
        selection1 = torch.unsqueeze(selection1,0)
        emb_anchor = selection1
        # emb_anchor = self.mini_unet.select_and_add_coordinates(emb_relative, anchor_coordinates)
        
        # selection2 = embedding[:,:,refernce_coordinates[:,:,1].long(),refernce_coordinates[:,:,0].long()].squeeze()
        if self.volume:
            selection2 = embedding[:,:,refernce_coordinates[:,:,0].long(),refernce_coordinates[:,:,1].long(),refernce_coordinates[:,:,2].long()].squeeze()
        else:
            selection2 = embedding[:,:,refernce_coordinates[:,:,1].long(),refernce_coordinates[:,:,0].long()].squeeze()
        
        if len(selection2.shape)==1:
            selection2 = selection2.unsqueeze(1)
        selection2 = selection2.transpose(1, 0)
        selection2[:,0] += refernce_coordinates[:,:,0].squeeze()
        selection2[:,1] += refernce_coordinates[:,:,1].squeeze()
        selection2[:,2] += refernce_coordinates[:,:,2].squeeze()
        emb_ref = selection2

        self.anchor_loss = AnchorLoss(self.temperature)
        anchor_loss = self.anchor_loss(emb_anchor, emb_ref)

        # self.log_images(x, emb_relative)

        self.log(
            'anchor_loss',
            anchor_loss.detach())
        
        self.last_loss = anchor_loss.detach()
        loss = anchor_loss

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        return optimizer

    # def log_images(self, raw, network_prediction):
    #     from pathlib import Path
    #     import numpy as np
    #     import math
    #     if self.global_step > 0 and (math.log2(self.global_step).is_integer() or self.global_step % 1000 == 0):
    #         Path("img").mkdir(exist_ok=True)
    #         x = raw.detach().cpu().numpy()

    #         if len(x.shape) == 5:
    #             x = np.transpose(x, (0, 1, 3, 2, 4))
    #             x = np.reshape(x, (-1, x.shape[-2]* x.shape[-1]))
    #             imsave(f"img/raw_ssl_{self.global_step}.png", x, check_contrast=False)
    #         elif len(x.shape) == 4:
    #             x = np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1))
    #             imsave(f"img/raw_ssl_{self.global_step}.png", x, check_contrast=False)

    #         x = network_prediction.detach().cpu().numpy()
    #         x = np.clip(x, -100, 100)
    #         if len(x.shape) == 5:
    #             x = x - np.mean(x, axis=(2,3,4), keepdims=True)
    #             x = x / np.max(np.abs(x), axis=(2, 3, 4), keepdims=True)
    #             x = np.transpose(x, (0, 1, 3, 2, 4))
    #             x = np.reshape(x, (-1, x.shape[-2] * x.shape[-1]))
    #             print(x.shape)
    #             imsave(f"img/pred_ssl_{self.global_step}.png", x, check_contrast=False)
    #         elif len(x.shape) == 4:
    #             x = np.reshape(np.transpose(x, (0, 2, 1, 3)),
    #                            (x.shape[0] * x.shape[1] * x.shape[2], -1))
    #             imsave(f"img/pred_ssl_{self.global_step}.png", x, check_contrast=False)

    #         model_directory = os.path.abspath(os.path.join(self.logger.log_dir,
    #                                                        os.pardir,
    #                                                        os.pardir,
    #                                                        "models"))

    #         if self.global_step > 100:
    #             os.makedirs(model_directory, exist_ok=True)
    #             model_save_path = os.path.join(
    #                 model_directory, f"mini_unet_{self.global_step:08d}_{self.local_rank:02}.torch")
    #             torch.save({"model_state_dict": self.mini_unet.state_dict()}, model_save_path)


    # def log_now(self, val=False):

    #     if val:
    #         if self.last_val_log == self.global_step:
    #             return False
    #         else:
    #             self.last_val_log = self.global_step
    #             return True

    #     if self.global_step > 1024:
    #         return self.global_step % 2048 == 0
    #     else:
    #         return self.global_step % 64 == 0
