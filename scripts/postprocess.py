from tqdm import trange
import zarr
import sys
from scipy.ndimage.morphology import distance_transform_edt as dtf
import numpy as np

inpfn = sys.argv[1]
inputkey = sys.argv[2]

z = zarr.open(inpfn, "r+")
growd = 3
th = 6

inkey = f"{inputkey}"
inlabels = z[inkey]
outkey = f"{inputkey}_postprocessed"

zout = z.create_dataset(outkey, shape=inlabels.shape, chunks=(16, -1, -1, -1), overwrite=True)
for t in trange(len(inlabels)):
    seg = inlabels[t, ..., 0].copy()
    dist = dtf(inlabels[t,...,0] == 0)
    mskt = dist < growd
    dist = dtf(mskt > 0)
    seg[dist < th] = 0
    zout[t, ..., 0] = seg
