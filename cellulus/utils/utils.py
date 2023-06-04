import numpy as np
import os
import urllib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import zarr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
            
def normalize_min_max_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def extract_data(zarr_url, data_dir, project_name):
    """
        Extracts data from `zip_url` to the location identified by `data_dir` parameters.

        Parameters
        ----------
        zarr_url: string
            Indicates the external url from where the data is downloaded
        data_dir: string
            Indicates the path to the directory where the data should be saved.
        Returns
        -------

    """
    if not os.path.exists(os.path.join(data_dir, project_name)):
        os.makedirs(data_dir)
        print("Created new directory {}".format(data_dir))
    
        with urlopen(zarr_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(data_dir)
        print("Downloaded and unzipped data to the location {}".format(data_dir))
    else:
        print("Directory already exists at the location {}".format(os.path.join(data_dir, project_name)))
              
def visualize(data_dir, train_val_dir='val', n_images=5, new_cmp='magma'):
    """
        Parameters
        -------
        data_dir: str
            Place where crops are (for example: data_dir='./')
        train_val_dir: str
            One of 'train' or 'val'
        n_images: int
            Number of columns
        new_cmp: Color Map

        Returns
        -------

    """
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    zarr_input = zarr.open(os.path.join(data_dir, train_val_dir + '.zarr'))
    indices = np.random.randint(0, zarr_input['raw'].shape[0], n_images)
    fig = plt.figure(constrained_layout=False, figsize=(16, 10))
    spec = gridspec.GridSpec(ncols=n_images, nrows=3, figure=fig)
    
    for i, index in enumerate(indices):
        im0 = zarr_input['raw'][index][...,0] 
        im1 = zarr_input['raw'][index][...,1]
        vmin0, vmax0 = np.min(im0), np.max(im0)
        vmin1, vmax1 = np.min(im1), np.max(im1)
        # TODO --> here the `visualize` is very specific to the tissuenet dataset
        ax0 = fig.add_subplot(spec[0, i])
        ax0.imshow(im0, cmap='magma', interpolation='None', vmin = vmin0, vmax = vmax0)
        ax0.axes.get_xaxis().set_visible(False)
        ax0.set_yticklabels([])
        ax0.set_yticks([])
        if i == 0:
            ax0.set_ylabel('IM-0', fontdict=font)
        ax1 = fig.add_subplot(spec[1, i])
        ax1.imshow(im1, cmap='magma', interpolation='None', vmin=vmin1, vmax = vmax1)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        if i == 0:
            ax1.set_ylabel('IM-1', fontdict=font)
    plt.tight_layout(pad=0, h_pad=0)
    plt.show()
    