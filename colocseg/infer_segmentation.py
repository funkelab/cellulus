import argparse
import sys

import numpy as np
import torch
import zarr

from colocseg.evaluation import *
from colocseg.inference import (affinity_segmentation, infer,
                                stardist_segmentation)
from colocseg.model import Unet2D

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("checkpoint")
parser.add_argument("dataset_file")
parser.add_argument("split")
parser.add_argument("out_channels", type=int)
parser.add_argument("valid_crop", type=int)
parser.add_argument("output_file")
parser.add_argument("min_size", type=int)
parser.add_argument("max_idx", type=int)
parser.add_argument("idx", type=int)
args = parser.parse_args()


model_state_dict = torch.load(args.checkpoint)["model_state_dict"]
model = Unet2D(2,
               args.out_channels,
               depth=3)
model.load_state_dict(model_state_dict)
model = model.cuda()

zin = zarr.open(args.dataset_file, "a")
asv = AnchorSegmentationValidation()

with torch.no_grad():

    raw = zin[f"{args.split}/raw"][args.idx:args.idx + 1]
    raw = np.transpose(raw, (0, 3, 1, 2))
    x = torch.from_numpy(raw).cuda()
    y = zin[f"{args.split}/gt"][args.idx:args.idx + 1]
    y = np.transpose(y, (0, 3, 1, 2))

    print(x.shape, y.shape)
    network_prediction = infer([x, y], model, args.valid_crop)
    print(network_prediction.shape)

    mws_segmentation_supervised = stardist_segmentation(
        raw, network_prediction, min_size=args.min_size)
    mws_segmentation_supervised = mws_segmentation_supervised[None]

    outzarr = zarr.open(args.output_file, "w")
    outzarr.require_dataset(f"raw",
                            shape=(args.max_idx, ) + raw.shape[1:],
                            chunks=raw.shape,
                            dtype=np.float32,
                            compression='gzip')[args.idx:args.idx + 1] = raw
    outzarr.require_dataset(f"gt", shape=(args.max_idx, ) + y.shape[1:],
                            chunks=y.shape,
                            dtype=np.int32,
                            compression='gzip')[args.idx:args.idx + 1] = y
    outzarr.require_dataset(f"prediction", shape=(args.max_idx, ) +
                            network_prediction.shape[1:],
                            dtype=np.float32,
                            chunks=network_prediction.shape,
                            compression='gzip')[args.idx:args.idx + 1] = network_prediction
    outzarr.require_dataset(f"segmentation", shape=(args.max_idx, ) +
                            mws_segmentation_supervised.shape[1:],
                            chunks=mws_segmentation_supervised.shape,
                            dtype=np.int32,
                            compression='gzip')[args.idx:args.idx + 1] = mws_segmentation_supervised
