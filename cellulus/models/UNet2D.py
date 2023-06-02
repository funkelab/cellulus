import torch
import torch.nn as nn
from funlib.learn.torch.models import UNet


class UNet2D(nn.Module):
    """
    A class used to create a PyTorch U-Net 2D Model.
    Attributes:
    _____________
    in_channels: int
                Number of channels in the input images (crops).
                Set to `1` for gray-scale images (e.g. images from cel tracking challenge)
                >1 for multi-channel images (e.g. set to `2` for images from the Tissue-Net dataset)
    out_channels: int
                Number of channels predicted epr pixel.
                Set to `2` by default.
                Setting to >3 would produce high-dimensional (non-spatial) embeddings
    aux_channels: int
                TODO
    num_fmaps: int
                TODO
    fmap_inc_factor: int
                TODO
    features_in_last_layer: int
                TODO
    head_type:  str
                TODO
    depth:      int
                Depth of the U-Net 2D model i.e. the number of rounds of down-sampling.
                Default = 3

    Methods
    _____________
    __init__:
                TODO
    head_forward:
                TODO
    get_absolute_embeddings:
                TODO
    forward:
                TODO
    """


    def __init__(
            self,
            in_channels,
            out_channels=2,
            aux_channels=0,
            num_fmaps=64,
            fmap_inc_factor=3,
            features_in_last_layer=64,
            head_type="single",
            depth=3):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.features_in_last_layer = features_in_last_layer
        self.head_type = head_type

        d_factors = [(2, 2), ] * depth
        self.backbone = UNet(in_channels=self.in_channels,
                             num_fmaps=num_fmaps,
                             fmap_inc_factor=fmap_inc_factor,
                             downsample_factors=d_factors,
                             activation='ReLU',
                             batch_norm=False,
                             padding='valid',
                             num_fmaps_out=self.features_in_last_layer,
                             kernel_size_down=[[(3, 3), (1, 1), (1, 1), (3, 3)]] * (depth + 1),
                             kernel_size_up=[[(3, 3), (1, 1), (1, 1), (3, 3)]] * depth,
                             constant_upsample=True)

        # Commonly used Non-linear projection head
        # see https://arxiv.org/pdf/1906.00910.pdf
        # or https://arxiv.org/pdf/2002.05709.pdf
        if head_type == "single":
            self.head = torch.nn.Sequential(nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer, 1),
                                            nn.ReLU(),
                                            nn.Conv2d(self.features_in_last_layer, out_channels + aux_channels, 1))
        elif head_type == "seq":
            self.head_pre = torch.nn.Sequential(nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer + aux_channels, 1))
            self.head_post = torch.nn.Sequential(nn.Conv2d(self.features_in_last_layer + aux_channels, self.features_in_last_layer, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(self.features_in_last_layer, out_channels, 1))
        elif head_type == "multi":
            self.head_main = torch.nn.Sequential(nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(self.features_in_last_layer, out_channels, 1))
            self.head_aux = torch.nn.Sequential(nn.Conv2d(self.features_in_last_layer, self.features_in_last_layer, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(self.features_in_last_layer, aux_channels, 1))

    def head_forward(self, last_layer_output):
        if self.head_type == "single":
            out_cat = self.head(last_layer_output)
            if self.aux_channels == 0:
                return out_cat
            else:
                return out_cat[:, self.out_channels:], out_cat[:, :self.out_channels]
        elif self.head_type == "seq":
            pre_out = self.head_pre(last_layer_output)
            out = self.head_post(pre_out)
            return pre_out[:, :self.aux_channels], out
        elif self.head_type == "multi":
            out = self.head_main(last_layer_output)
            out_aux = self.head_aux(last_layer_output)
            return out_aux, out


    @staticmethod
    def get_absolute_embeddings(predicted_embeddings, coordinates):
        """
        A method used to add the predicted, relative embeddings (for e.g. the spatial offsets or OCEs) to the
        absolute locations of the pixel (voxel) coordinates.

        Attributes
        _____________
        predicted_embeddings: shape is B, 2, H, W
        TODO
        coordinates: shape is B, N, 2
        TODO
        Returns
        _____________
        TODO
        """

        absolute_embeddings = []
        for predicted_embedding, coordinate in zip(predicted_embeddings, coordinates):
            absolute_embedding = predicted_embedding[:, coordinate[:, 1], coordinate[:, 0]]
            absolute_embedding = absolute_embedding.transpose(1, 0)
            absolute_embedding += coordinate
            absolute_embeddings.append(absolute_embedding)

        return torch.stack(absolute_embeddings, dim=0)

    def forward(self, raw_images):
        """
        A method used to feed in the batch of raw image patches.

        Attributes
        _____________
        raw_images: shape is B, C, H, W
                    where C is the number of channels in the raw images
                    (for example, C = 1 for gray-scale images)

        Returns
        _____________
        TODO
        """
        h = self.backbone(raw_images)
        return self.head_forward(h)

