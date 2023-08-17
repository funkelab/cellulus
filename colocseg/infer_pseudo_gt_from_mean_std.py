from scipy.ndimage import gaussian_filter
from colocseg.evaluation import segment_with_meanshift
from colocseg.utils import zarr_append, remove_border, sizefilter
from colocseg.evaluation import evaluate_predicted_zarr
from colocseg.evaluation import *
import zarr
import numpy as np
import torch
import sys
import stardist
from colocseg.utils import remove_border
from skimage.filters import threshold_otsu

# load miniunet
result_dataset = sys.argv[1]
input_dataset = sys.argv[2]
input_name = sys.argv[3]
result_name = sys.argv[4]
t = int(sys.argv[5])

th = sys.argv[6]
if th != "otsu":
    th = float(th)

zin = zarr.open(result_dataset, "r")
zout = zarr.open(result_dataset, "a")

def msseg(emb, raw, ac, bw, th=0.21):
    emb = torch.from_numpy(emb)
    emb[:, 1] += torch.arange(emb.shape[2])[None, :, None]
    emb[:, 0] += torch.arange(emb.shape[3])[None, None, :]

    if th == "otsu":
        mask = (threshold_otsu(ac) > ac)
        mask = mask[None]
    elif th == 0:
        mask = None
    else:
        ac = ac.copy()
        ac -= ac.min()
        ac /= ac.max()
        mask = (th > ac)
        mask = mask[None]
    
    seg = segment_with_meanshift(emb,
                                 bw,
                                 mask=mask,
                                 reduction_probability=0.1,
                                 cluster_all=False)[0]
    seg = stardist.fill_label_holes(seg)
    seg = remove_border(raw[..., 0],
                        raw[..., 1],
                        seg[None])[0]
    seg = sizefilter(seg, 10)
    return seg


if __name__ == "__main__":


        ameanstd = zin[input_name]
        with np.load(input_dataset) as data:
            raw = data['X'][t:t + 1]

        ac = ameanstd[t, 2]
        bw = 7
        emb = ameanstd[t:t + 1, :2]
        meanmeanshift_seg = msseg(emb, raw, ac, bw, th=th)
        w, h = meanmeanshift_seg.shape

        out_key = result_name
        outarr = zout.require_dataset(out_key,
                                    shape=(raw.shape[0], w, h, 1),
                                    chunks=(1, w, h, 1),
                                    compression='gzip',
                                    dtype=np.int32)
        outarr[t, ..., 0] = meanmeanshift_seg
