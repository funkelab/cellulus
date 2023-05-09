from skimage.filters import threshold_otsu
from colocseg.evaluation import *
import zarr
import numpy as np
import torch
import sys
from colocseg.model import Unet2D
from colocseg.utils import remove_border
from skimage.io import imsave

# load miniunet
checkpoint = sys.argv[1]
dataset_file = sys.argv[2]
result_name = sys.argv[3]
split = sys.argv[4]
idx = int(sys.argv[5])

model_state_dict = torch.load(checkpoint)["model_state_dict"]
model = Unet2D(2, 2, num_fmaps=256, features_in_last_layer=32)
model.load_state_dict(model_state_dict)
model = model.cuda()

for p_salt in [0.01]:

    for salt_value in [0.001, 0.01, 0.1, 0.2, 1.0]:

        zin = zarr.open(dataset_file, "a")

        asv = AnchorSegmentationValidation()
        ms_bandwidths = (7, )

        with torch.no_grad():
            x = zin[f"{split}/raw"][idx:idx + 1]
            x = np.transpose(x, (0, 3, 1, 2))
            x = np.pad(x, ((0, 0), (0, 0), (8, 8), (8, 8)), mode='constant')

            clean_input = torch.from_numpy(x.astype(np.float32)).cuda()
            predictions = []
            for _ in range(32):  # np.linspace(0.,1.,num=32):
                noisy_input = clean_input.detach().clone()
                rnd = torch.rand(*noisy_input.shape).cuda()
                noisy_input[rnd <= p_salt] = salt_value
                pred = model(noisy_input)[0].detach().cpu()
                predictions.append(pred)

            emb_std, emb = torch.std_mean(torch.stack(predictions, dim=0), dim=0, keepdim=False, unbiased=False)
            emb_std = emb_std.sum(dim=0, keepdim=True)
            emb_out = torch.cat((emb, emb_std), dim=0).numpy()
            c, w, h = emb_out.shape
            
            thresh = threshold_otsu(emb_out[2])
            imsave(
                f"/groups/funke/home/wolfs2/local/data/tissuenet/plot_data/tmp/mask_{idx}_{salt_value:0.3f}.png", emb_out[2] < thresh)
            
            imsave(f"/groups/funke/home/wolfs2/local/data/tissuenet/plot_data/tmp/emba_{idx}_{salt_value:0.3f}.png", emb_out[0])
            # imsave(
                # f"/groups/funke/home/wolfs2/local/data/tissuenet/plot_data/tmp/embb_{idx}_{salt_value:0.3f}.png", emb_out[1])
            imsave(
                f"/groups/funke/home/wolfs2/local/data/tissuenet/plot_data/tmp/embc_{idx}_{salt_value:0.3f}.png", emb_out[2])
            # if f'{split}/{result_name}' not in zin:
            #     zin.create_dataset(f'{split}/{result_name}',
            #                     shape=(zin[f"{split}/raw"].shape[0], c, w, h),
            #                     chunks=(1, c, w, h),
            #                     compression='gzip',
            #                     dtype=np.float32)
            # zin[f'{split}/{result_name}'][idx] = emb_out
            

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
