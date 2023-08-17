import torch
import torch.nn as nn
from funlib.learn.torch.models import UNet


class Unet2D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            aux_channels=0,
            num_fmaps=64,
            fmap_inc_factor=3,
            features_in_last_layer=64,
            head_type="single",
            depth=1):

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
    def select_and_add_coordinates(output, coords):
        selection = []
        # output.shape = (b, c, h, w)
        for o, c in zip(output, coords):
            sel = o[:, c[:, 1], c[:, 0]]
            sel = sel.transpose(1, 0)
            sel += c
            selection.append(sel)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selection, dim=0)

    def forward(self, raw):
        h = self.backbone(raw)
        return self.head_forward(h)

    def forward_and_select(self, raw, coords):
        # coords.shape = (b, p, 2)
        h = self.forward(raw)
        return self.select_coords(h, coords)
