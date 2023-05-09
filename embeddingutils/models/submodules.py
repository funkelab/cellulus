from embeddingutils.affinities import offset_slice, offset_padding, get_offsets
from inferno.extensions.layers.convolutional import ConvELU3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import cat
import numpy as np

try:
    from speedrun.log_anywhere import log_image
except ImportError:
    def log_image(tag, value):
        assert False, f'Image logging cannot be used without speedrun.'

try:
    from gpushift import MeanShift
except ImportError:
    class MeanShift:
        def __init__(self, *args, **kwargs):
            assert False, f'gpushift not found. please install from https://github.com/imagirom/gpushift.'


class DepthToChannel(nn.Module):
    def forward(self, input_):
        assert len(input_.shape) == 5, \
            f'input must be 5D tensor of shape (B, C, D, H, W), but got shape {input_.shape}.'
        input_ = input_.permute((0, 2, 1, 3, 4))
        return input_.contiguous().view((-1, ) + input_.shape[-3:])


class Normalize(nn.Module):
    def __init__(self, dim=1):
        super(Normalize, self).__init__()
        self.dim=dim

    def forward(self, input_):
        return F.normalize(input_, dim=self.dim)


class ResBlock(nn.Module):
    def __init__(self, inner, pre=None, post=None, outer=None):
        super(ResBlock, self).__init__()
        self.inner = inner
        self.pre = pre
        self.post = post
        self.outer = outer

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        if hasattr(self, 'outer') and self.outer is not None:
            skip = self.outer(x)
        else:
            skip = x
        x = skip + self.inner(x)
        if self.post is not None:
            x = self.post(x)
        return x


class ValidPadResBlock(ResBlock):
    def __init__(self, f_in, f_main=None, kernel_size=1, conv_type=nn.Conv3d, activation='ELU'):
        f_main = f_in if f_main is None else f_main
        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)()
        inner = nn.Sequential(
            conv_type(f_in, f_main, kernel_size=1),
            activation,
            conv_type(f_main, f_main, kernel_size=kernel_size, padding=0),
            activation,
            conv_type(f_main, f_in, kernel_size=1),
        )
        self.crop = (kernel_size - 1)//2
        super(ValidPadResBlock, self).__init__(inner=inner, outer=self.outer)

    def outer(self, x):
        crop = self.crop
        if crop == 0:
            return x
        else:
            return x[:, :, slice(crop, -crop), slice(crop, -crop), slice(crop, -crop)]


class SuperhumanSNEMIBlock(ResBlock):
    def __init__(self, f_in, f_main=None, f_out=None,
                 pre_kernel_size=(1, 3, 3), inner_kernel_size=(3, 3, 3),
                 conv_type=ConvELU3D):
        if f_main is None:
            f_main = f_in
        if f_out is None:
            f_out = f_main
        pre = conv_type(f_in, f_out, kernel_size=pre_kernel_size)
        inner = nn.Sequential(conv_type(f_out, f_main, kernel_size=inner_kernel_size),
                              conv_type(f_main, f_out, kernel_size=inner_kernel_size))
        super(SuperhumanSNEMIBlock, self).__init__(pre=pre, inner=inner)


"""modified convgru implementation of https://github.com/jacobkimmel/pytorch_convgru"""
class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, conv_type):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # padding = kernel_size // 2
        hs = hidden_size
        self.reset_gate = conv_type(input_size + hs, hs, kernel_size)
        self.update_gate = conv_type(input_size + hs, hs, kernel_size)
        self.out_gate = conv_type(input_size + hs, hs, kernel_size)

        # init.orthogonal(self.reset_gate.weight)
        # init.orthogonal(self.update_gate.weight)
        # init.orthogonal(self.out_gate.weight)
        # init.constant(self.reset_gate.bias, 0.)
        # init.constant(self.update_gate.bias, 0.)
        # init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = input_.new(np.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_sizes, n_layers, conv_type):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_size : integer. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if isinstance(kernel_sizes, (list, tuple)):
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes]*n_layers

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            cell = ConvGRUCell(self.input_size, self.hidden_size, self.kernel_sizes[i], conv_type)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length

    def forward(self, input_, hidden=None):
        '''
        Parameters
        ----------
        x : 5D/6D input tensor. (batch, channels, sequence, height, width, *depth)).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        # warmstart Gru hidden state with one pass through layer 0

        sl = self.sequence_length
        upd_hidden = []
        for batch in range(input_.shape[0] // self.sequence_length):
            time_index = batch * sl

            for idx in range(self.n_layers):
                upd_cell_hidden = self.cells[idx](input_[time_index:time_index + 1], None).detach()

            for s in range(self.sequence_length):
                x = input_[time_index + s:time_index + s + 1]
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    # pass through layer
                    upd_cell_hidden = cell(x, upd_cell_hidden)

                upd_hidden.append(upd_cell_hidden)

        # retain tensors in list to allow different hidden sizes
        return cat(upd_hidden, dim=0)


class ShakeShakeFn(torch.autograd.Function):
    # modified from https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        # first dim is assumed to be batch
        if training:
            alpha = torch.rand(x1.size(0), *((1,)*(len(x1.shape)-1)), dtype=x1.dtype, device=x1.device)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.rand(grad_output.size(0), *((1,) * (len(grad_output.shape) - 1)),
                          dtype=grad_output.dtype, device=grad_output.device)

        return beta * grad_output, (1 - beta) * grad_output, None


class ShakeShakeMerge(nn.Module):
    def forward(self, x1, x2):
        return ShakeShakeFn.apply(x1, x2, self.training)


class SampleChannels(nn.Module):
    def __init__(self, n_selected_channels):
        super(SampleChannels, self).__init__()
        self.n_selected_channels = n_selected_channels

    def sample_ind(self, n_channels):
        assert self.n_selected_channels <= n_channels
        result = np.zeros(n_channels)
        result[np.random.choice(np.arange(n_channels), self.n_selected_channels, replace=False)] = 1
        return result

    def forward(self, input):
        n_channels = input.size(1)
        ind = np.stack([self.sample_ind(n_channels) for _ in range(input.size(0))])
        ind = torch.ByteTensor(ind).to(input.device)
        return input[ind].view(input.size(0), self.n_selected_channels, *input.shape[2:])


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, input):
        return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)


class AffinityBasedAveraging(torch.nn.Module):
    def __init__(self, offsets, extra_dims=2, softmax=True, activation=None, normalize=True, **pad_kwargs):
        super(AffinityBasedAveraging, self).__init__()
        self.pad_kwargs = dict(mode='replicate', **pad_kwargs)
        self.offsets = get_offsets(offsets)
        self.offset_slices = [offset_slice(off, extra_dims=extra_dims) for off in self.offsets]
        self.reverse_offset_slices = [offset_slice(-off, extra_dims=extra_dims) for off in self.offsets]
        self.offset_padding = [offset_padding(off) for off in self.offsets]
        self.use_softmax = softmax
        if self.use_softmax:
            assert activation is None, f'activation function is overriden by using softmax!'
        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)()
        self.activation = activation
        self.normalize = normalize

    def forward(self, affinities, embedding):
        padded_embeddings = []
        for sl, pad in zip(self.offset_slices, self.offset_padding):
            padded_embeddings.append(F.pad(embedding[sl], pad, **self.pad_kwargs))
        padded_embeddings = torch.stack(padded_embeddings, dim=1)
        if self.use_softmax:
            affinities = F.softmax(affinities, dim=1)
        elif hasattr(self, 'activation') and self.activation:
            affinities = self.activation(affinities)
        if hasattr(self, 'normalize') and self.normalize:
            affinities = F.normalize(affinities, dim=1, p=1)
        counts = affinities.new_zeros((affinities.shape[0], 1) + affinities.shape[2:])
        return (padded_embeddings * affinities[:, :, None]).sum(1)


class HierarchicalAffinityAveraging(torch.nn.Module):
    def __init__(self, levels=2, dim=2, stride=1, append_affinities=False, ignore_n_first_channels=0, log_images=False,
                 **kwargs):
        """ averages iteratively with thrice as long offsets in every level """
        super(HierarchicalAffinityAveraging, self).__init__()

        self.base_neighborhood = stride * np.mgrid[dim*(slice(-1, 2),)].reshape(dim, -1).transpose()
        self.stages = nn.ModuleList([AffinityBasedAveraging(3**i * self.base_neighborhood, **kwargs)
                                     for i in range(levels)])
        self.levels = levels
        self.dim = dim
        self.append_affinities = append_affinities
        self.ignore_n_first_channels = ignore_n_first_channels
        self.log_images = log_images

    def forward(self, input):
        ignored = input[:, :self.ignore_n_first_channels]
        input = input[:, self.ignore_n_first_channels:]

        affinity_groups = input[:, :len(self.base_neighborhood) * self.levels]
        affinity_groups = affinity_groups.reshape(
            input.size(0), self.levels, len(self.base_neighborhood), *input.shape[2:])\
            .permute(1, 0, *range(2, 3 + self.dim))
        embedding = input[:, len(self.base_neighborhood) * self.levels:]
        for i, (affinities, stage) in enumerate(zip(affinity_groups, self.stages)):
            if self.log_images:
                log_image(f'embedding_stage_{i}', embedding)
                log_image(f'affinities_stage_{i}', affinities)

            embedding = stage(affinities, embedding)
        return torch.cat([ignored, embedding], 1)


class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class MeanShiftLayer(MeanShift):
    """
    Wrapper for MeanShift that handles appropriate reshaping.
    """
    def forward(self, embedding):
        in_shape = embedding.shape  # B E (D) H W
        embedding = embedding.view(in_shape[:2] + (-1,))  # B E N
        embedding = embedding.transpose(1, 2)  # B N E
        embedding = super(MeanShiftLayer, self).forward(embedding)
        embedding = ContiguousBackward().apply(embedding)
        embedding = embedding.transpose(1, 2)  # B E N
        embedding = embedding.view(in_shape)  # B E (D) H W
        return embedding


class ChannelSliceWrapper(torch.nn.Module):
    """
    Wrapper to apply a module only to some channels.
    """
    def __init__(self, module, start=0, stop=None):
        super(ChannelSliceWrapper, self).__init__()
        self.slice = slice(start, stop)
        self.module = module

    def forward(self, input):
        input[:, self.slice] = self.module(input[:, self.slice])
        return input


if __name__ == '__main__':

    model = AffinityBasedAveraging(offsets=np.array([[0, 1], [1, 0]]))
    emb = torch.tensor([[[
        [0, 1],
        [2, 3]
    ]]]).float()
    aff = torch.tensor([[
        [[-10, 0],
         [0, 0]],
        [[0, 0],
         [0, 0]],
    ]]).float()
    print(emb.shape)
    print(aff.shape)
    out = model(aff, emb)
    print('out: ', out)
    print(out.shape)

    assert False


    from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D

    a = torch.rand(3, 3).requires_grad_(True)
    model = SampleChannels(2)

    print(a)
    b = model(a)
    print(b)
    b.sum().backward()
    print(a.grad)
    assert False



    model = ConvGRU(4, 8, (3, 5, 3), 3, Conv2D)

    print(model)
    model = model.cuda()
    model.set_sequence_length(8)
    shape = tuple((16, 1, 100, 100))
    inp = torch.ones(shape).cuda()
    out = model(inp)
    print(inp.shape)
    print(out.shape)
