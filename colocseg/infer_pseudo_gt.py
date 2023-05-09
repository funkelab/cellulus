# from colocseg.evaluation import *
import zarr
import numpy as np
import torch
import sys
from colocseg.model import Unet2D

# load miniunet
checkpoint = sys.argv[1]
dataset_file = sys.argv[2]
result_name = sys.argv[3]
split = sys.argv[4]
idx = int(sys.argv[5])

if len(sys.argv) > 6:
    input_key = sys.argv[6]
else:
    input_key = "raw"

if len(sys.argv) > 7:
    in_channels = int(sys.argv[7])
else:
    in_channels = 2

transpose = True
if len(sys.argv) > 8:
    transpose = sys.argv[8] == "transpose"
    
model_state_dict = torch.load(checkpoint)["model_state_dict"]
model = Unet2D(in_channels, 2, num_fmaps=256)
model.load_state_dict(model_state_dict)
model = model.cuda()

p_salt = 0.01
zin = zarr.open(dataset_file, "a")
split_limit = 800
step = 120
ms_bandwidths = (7, )

with torch.no_grad():
    x = zin[f"{split}/{input_key}"][idx:idx + 1]
    if transpose:
        x = np.transpose(x, (0, 3, 1, 2))
    x = np.pad(x, ((0, 0), (0, 0), (8, 8), (8, 8)), mode='constant')

    clean_input = torch.from_numpy(x.astype(np.float32)).cuda()
    
    predictions = []
    for salt_value in [0.5, 1.0]:
        for _ in range(16):
            noisy_input = clean_input.detach().clone()    
            rnd = torch.rand(*noisy_input.shape).cuda()
            noisy_input[rnd <= p_salt] = salt_value

            if x.shape[-1] > split_limit:
                pred = []
                for idx_low in range(8, x.shape[-1] - 8, step):
                    inp = noisy_input[..., idx_low - 8:idx_low + step + 8]
                    pred.append(model(inp)[0].detach().cpu())
                pred = torch.cat(pred, dim=-1)
            else:
                pred = model(noisy_input)[0].detach().cpu()
            predictions.append(pred)

        emb_std, emb = torch.std_mean(torch.stack(predictions, dim=0), dim=0, keepdim=False, unbiased=False)
        emb_std = emb_std.sum(dim=0, keepdim=True)
        emb_out = torch.cat((emb, emb_std), dim=0)
        c, w, h = emb_out.shape
        key = f'{split}/{result_name}_{salt_value:0.1f}'
        out_ds = zin.require_dataset(key,
                                    shape=(zin[f"{split}/{input_key}"].shape[0], c, w, h),
                                    chunks=(1, c, w, h),
                                    compression='gzip',
                                    dtype=np.float32)
        out_ds[idx] = emb_out

    # emb[:, 1] += torch.arange(emb.shape[2])[None, :, None]
    # emb[:, 0] += torch.arange(emb.shape[3])[None, None, :]
    # seg = asv.meanshift_segmentation(emb, ms_bandwidths)[ms_bandwidths[0]][0]
    # seg += 1
    # seg = remove_border(zin[f"{split}/raw"][idx:idx + 1, ..., 0],
    #                     zin[f"{split}/raw"][idx:idx + 1, ..., 1], seg[None])[0]
    # w, h = seg.shape[-2:]
    # if f'{split}/{result_name}' not in zin:
    #     zin.create_dataset(f'{split}/{result_name}',
    #                        shape=(zin[f"{split}/raw"].shape[0], w, h, 1),
    #                        chunks=(1, w, h, 1),
    #                        compression='gzip',
    #                        dtype=np.int32)
    # zin[f'{split}/{result_name}'][idx, ..., 0] = seg
