import numpy as np
import zarr
import sys
import os

tissuenet_folder = sys.argv[1]

output_file = os.path.join(tissuenet_folder, "tissuenet_v1.0.zarr")
zout = zarr.open(output_file, "w")
splits = {"train": "tissuenet_v1.0_train.npz",
          "val": "tissuenet_v1.0_val.npz",
          "test": "tissuenet_v1.0_test.npz"}

for split, fn in splits.items():
    with np.load(os.path.join(tissuenet_folder, fn)) as data:
        raw_data = data['X']
        gt_data = data['y']
        w, h = raw_data.shape[1:3]
        zout.create_dataset(f"{split}/raw", data=raw_data, chunks=(1, w, h, 1), compression='gzip')
        zout.create_dataset(f"{split}/gt", data=gt_data, chunks=(1, w, h, 1), compression='gzip', dtype="int32")
