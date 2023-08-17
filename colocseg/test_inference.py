import sys

from argparse import ArgumentParser
from colocseg.utils import zarr_append
from tqdm import tqdm
import zarr

from colocseg.inference import infer, stardist_segmentation, cellpose_segmentation
import torch
from glob import glob
from colocseg.model import Unet2D
import shlex
import pandas as pd
from colocseg.evaluation import evaluate_predicted_zarr
import pytorch_lightning as pl
from colocseg.datasets import TissueNetDataset
from argparse import ArgumentParser

tissuenet_test_dataset = "path_to_ds/tissuenet_v1.0_test.npz"

def test_from_checkpoint(root_folder, args, checkpoint_index="all", target_type="cell", tissue_type="immune"):

    cp_file = glob(f"{root_folder}/models/maxi_*.torch")
    cp_file.sort(reverse=True)
    if checkpoint_index != "all":
        idx = int(checkpoint_index)
        print("testing model index ", idx, " of ", cp_file)
        if idx == -1:
            cp_file = cp_file[-1:]
        else:
            cp_file = cp_file[idx:idx + 1]
        print(cp_file)

    for checkpoint in cp_file:
        model_state_dict = torch.load(checkpoint)["model_state_dict"]
        model = Unet2D(args.in_channels,
                    args.out_channels,
                    args.aux_channels,
                    head_type=args.unet_head_type,
                    fmap_inc_factor=args.unet_fmap_inc_factor,
                    num_fmaps=32,
                    depth=3,
                    )
        model.load_state_dict(model_state_dict)
        model = model.cuda().eval()

        ds = TissueNetDataset(
            tissuenet_test_dataset,
            augment=False,
            tissue_type=tissue_type,
            target_type=target_type,
            )
        ds.valid_crop = 0

        tissue_type_name = tissue_type if tissue_type is not None else "all"

        iteration = int(checkpoint.split("_")[-2])
        zout_file = f"{root_folder}/test_{target_type}_{tissue_type_name}_{iteration}.zarr"
        zarr.open(zout_file, "w")

        for i, batch in tqdm(enumerate(ds)):

            with torch.no_grad():
                batch = torch.from_numpy(batch[0]).cuda()[None], torch.from_numpy(batch[1]).cuda()[None]

                network_prediction = infer(batch, model, args.valid_crop+1)
                raw = batch[0].cpu().numpy()

                if args.loss_name_super == 'StardistLoss':
                    network_prediction[:, 0] = torch.from_numpy(network_prediction[:, 0]).sigmoid().numpy()
                elif args.loss_name_super == 'CellposeLoss':
                    network_prediction[:, 3] = torch.from_numpy(network_prediction[:, 3]).sigmoid().numpy()                
                
                outzarr = zarr.open(zout_file, "a")
                zarr_append("gt", batch[1][None].cpu().numpy(), outzarr)
                if i < 4:
                    zarr_append("raw", raw, outzarr)
                    zarr_append("prediction", network_prediction, outzarr)
                    
                if args.loss_name_super == 'StardistLoss':
                    for prob_thresh in [0.3, 0.4, 0.486166]:
                        sd_segmentation_supervised = stardist_segmentation(
                            raw, network_prediction, min_size=30, prob_thresh=prob_thresh)

                        if prob_thresh == 0.486166:
                            zarr_append("segmentation", sd_segmentation_supervised[None], outzarr, attr=(
                                "model", checkpoint, "th", prob_thresh))
                        else:
                            zarr_append(f"segmentation_{prob_thresh:0.3f}",
                                        sd_segmentation_supervised[None], outzarr, attr=("model", checkpoint, "th", prob_thresh))
                elif args.loss_name_super == 'CellposeLoss':
                    sd_segmentation_supervised = cellpose_segmentation(
                           raw, network_prediction, min_size=30)
                    zarr_append("segmentation", sd_segmentation_supervised[None], outzarr, attr=(
                                "model", checkpoint, "method", "cellpose"))
                else:
                    raise NotImplementedError("loss name not recognized")
                    
        for k in [k for k in zarr.open(zout_file, "r").keys() if k.startswith("segmentation")]:
            scores = evaluate_predicted_zarr(
                zout_file, f"{root_folder}/test_scores_{target_type}_{tissue_type_name}_individual_{iteration}_{k}.csv",
                seg_key=k)

            outdict = {key: value for key, value in vars(args).items() if key[:1] != "_"}
            outdict.update(scores.to_dict())
            outdict["iteration"] = iteration
            outdict["tissue_type_test"] = tissue_type

            df = pd.DataFrame.from_dict(outdict, orient="index")
            df.T.to_csv(f"{root_folder}/test_data_{target_type}_{tissue_type_name}_{iteration}_{k}.csv")


if __name__ == "__main__":

    root_folder = sys.argv[1]
    sys.path.append(root_folder)
    from colocseg.trainingmodules import PartiallySupervisedTrainer
    from colocseg.datamodules import PartiallySupervisedDataModule

    if sys.argv[3] != "alltypes":
        tissue_type = sys.argv[3]
        tissue_type_name = tissue_type
    else:
        tissue_type = None
        tissue_type_name = "alltypes"

    target_type = sys.argv[4]

    parser = ArgumentParser()
    parser = PartiallySupervisedTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PartiallySupervisedDataModule.add_argparse_args(parser)
    parser = PartiallySupervisedDataModule.add_model_specific_args(parser)

    sc_file = f"{root_folder}/train.sh"

    with open(sc_file, 'r') as f:
        lines = f.read().splitlines()
        call_string_pre_split_eq = shlex.split(lines[-1])[2:]

    args = parser.parse_args(call_string_pre_split_eq)
    print(args.__dict__.items())
    checkpoint_index = sys.argv[2]

    test_from_checkpoint(root_folder, args, checkpoint_index)
