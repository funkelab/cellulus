import os

import numpy as np
import torch
import zarr
from tqdm import tqdm

from cellulus.criterions.stardist_loss import StardistLoss
from cellulus.datasets import get_dataset
from cellulus.models import get_model
from cellulus.utils import get_logger


def semisupervised_train(semi_sup_exp_config):
    print(semi_sup_exp_config)

    if not os.path.exists("semi_supervised_models"):
        os.makedirs("semi_supervised_models")

    model_config = semi_sup_exp_config.model_config

    semi_sup_train_config = semi_sup_exp_config.semi_sup_train_config

    raw_dataset = get_dataset(
        dataset_config=semi_sup_train_config.raw_data_config,
        crop_size=tuple(semi_sup_train_config.crop_size),
        control_point_spacing=semi_sup_train_config.control_point_spacing,
        control_point_jitter=semi_sup_train_config.control_point_jitter,
        pseudo_dataset_config=semi_sup_train_config.pseudo_data_config,
        supervised_dataset_config=semi_sup_train_config.supervised_data_config,
        semi_supervised=True,
    )

    raw_dataloader = torch.utils.data.DataLoader(
        dataset=raw_dataset,
        batch_size=semi_sup_train_config.batch_size,
        drop_last=True,
        num_workers=semi_sup_train_config.num_workers,
        pin_memory=True,
    )

    model = get_model(
        in_channels=raw_dataset.get_num_channels(),
        out_channels=17,
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=[
            tuple(factor) for factor in model_config.downsampling_factors
        ],
        num_spatial_dims=raw_dataset.get_num_spatial_dims(),
    )

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = StardistLoss()

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=semi_sup_train_config.initial_learning_rate,
    )

    def lambda_(iteration):
        return pow((1 - ((iteration) / semi_sup_train_config.max_iterations)), 0.9)

    # set logger
    logger = get_logger(keys=["train"], title="loss")

    # resume training
    start_iteration = 0
    lowest_loss = 1e7

    if model_config.checkpoint is None:
        pass
    else:
        print(f"Resuming model from {model_config.checkpoint}")
        state = torch.load(model_config.checkpoint)
        start_iteration = state["iteration"] + 1
        lowest_loss = state["lowest_loss"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    # call `train_iteration`
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, semi_sup_train_config.max_iterations), raw_dataloader
        )
    ):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
        )

        train_loss, prediction = train_iteration(
            batch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        scheduler.step()
        logger.add(key="train", value=train_loss.cpu().detach().numpy())
        logger.write()
        logger.plot()

        if (iteration + 1) % semi_sup_train_config.save_model_every == 0:
            is_lowest = train_loss < lowest_loss
            lowest_loss = min(train_loss, lowest_loss)
            state = {
                "iteration": iteration,
                "lowest_loss": lowest_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
            save_model(state, iteration, is_lowest)

        if (iteration + 1) % semi_sup_train_config.save_snapshot_every == 0:
            save_snapshot(
                batch,
                prediction,
                iteration,
            )


def train_iteration(batch, model, criterion, optimizer, alpha=0.5):
    model.train()

    def unqiue_values_to_unique_ints(array):
        if np.any(array - array.astype(np.int16)):
            mapping = {v: k for k, v in enumerate(np.unique(array))}
            u, inv = np.unique(array, return_inverse=True)
            Y1 = np.array([mapping[x] for x in u])[inv].reshape(array.shape)
            array = Y1.astype(np.int16)
        return array

    if torch.cuda.is_available():
        batch["raw"] = batch["raw"].to("cuda")
        batch["pseudo_stardist"] = batch["pseudo_stardist"].to("cuda")
        batch["supervised_stardist"] = batch["supervised_stardist"].to("cuda")
        batch["pseudo_labels"] = batch["pseudo_labels"].to("cuda")
        batch["supervised_labels"] = batch["supervised_labels"].to("cuda")
        batch["supervised_labels"] = torch.tensor(
            unqiue_values_to_unique_ints(
                batch["supervised_labels"].cpu().detach().numpy()
            )
        ).to("cuda")

    prediction = model(batch["raw"])

    def combine_GT_and_pseudo_labels(gt_labels, pseudo_labels):
        if gt_labels.shape != pseudo_labels.shape:
            print("labelled images are different sizes")

        combined_labels = np.zeros(gt_labels.shape)
        combined_labels[:] = gt_labels.cpu().detach().numpy()[:]

        for pseudo_label_value in np.unique(pseudo_labels.cpu().detach().numpy()):
            if pseudo_label_value == 0.0:
                pass
            # for each pseudo label, check it does not intersect with a GT label.
            # if it doesn't intersect, add it to the combined labels.
            # if it does intersect, leave it or add to ignore mask?

            # all of the indicies that have this given label value
            # this_pseudo_label = np.where(np.any(pseudo_labels==pseudo_label_value))

            if np.any(
                gt_labels.cpu()
                .detach()
                .numpy()[pseudo_labels.cpu().detach().numpy() == pseudo_label_value]
            ):
                pass
            else:
                combined_labels = (
                    combined_labels
                    + pseudo_labels.cpu().detach().numpy()
                    * (pseudo_labels.cpu().detach().numpy() == pseudo_label_value)
                )
        return combined_labels

    use_combined_loss = True

    if use_combined_loss:
        from cellulus.criterions import stardist_transform

        gt_labels = batch["supervised_labels"][
            :, :, : prediction.shape[2], : prediction.shape[3]
        ]
        pseudo_labels = batch["pseudo_labels"][
            :, :, : prediction.shape[2], : prediction.shape[3]
        ]
        combined_labels = combine_GT_and_pseudo_labels(gt_labels, pseudo_labels)

        combined_stardist = (
            torch.tensor(stardist_transform(combined_labels)).cuda().unsqueeze(0)
        )

        loss = criterion(prediction, combined_stardist)

    else:
        supervised_loss = criterion(
            prediction,
            batch["supervised_stardist"][
                :, :, : prediction.shape[2], : prediction.shape[3]
            ],
        )
        semisupervised_loss = criterion(
            prediction,
            batch["pseudo_stardist"][
                :, :, : prediction.shape[2], : prediction.shape[3]
            ],
        )
        loss = (alpha * supervised_loss) + ((1 - alpha) * semisupervised_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, prediction


def save_model(state, iteration, is_lowest=False):
    file_name = os.path.join("semi_supervised_models", str(iteration).zfill(6) + ".pth")
    torch.save(state, file_name)
    print(f"Checkpoint saved at iteration {iteration}")
    if is_lowest:
        file_name = os.path.join("semi_supervised_models", "best_loss.pth")
        torch.save(state, file_name)


def save_snapshot(batch, prediction, iteration):
    num_spatial_dims = len(batch["raw"].shape) - 2

    axis_names = ["s", "c"] + ["t", "z", "y", "x"][-num_spatial_dims:]
    prediction_offset = tuple(
        (a - b) / 2
        for a, b in zip(
            batch["raw"].shape[-num_spatial_dims:], prediction.shape[-num_spatial_dims:]
        )
    )
    f = zarr.open("semi_supervised_snapshots.zarr", "a")
    f[f"{iteration}/raw"] = batch["raw"].detach().cpu().numpy()
    f[f"{iteration}/raw"].attrs["axis_names"] = axis_names

    f[f"{iteration}/pseudo_stardist"] = batch["pseudo_stardist"].detach().cpu().numpy()
    f[f"{iteration}/pseudo_stardist"].attrs["axis_names"] = axis_names

    f[f"{iteration}/pseudo_labels"] = batch["pseudo_labels"].detach().cpu().numpy()
    f[f"{iteration}/pseudo_labels"].attrs["axis_names"] = axis_names

    f[f"{iteration}/supervised_stardist"] = (
        batch["supervised_stardist"].detach().cpu().numpy()
    )
    f[f"{iteration}/supervised_stardist"].attrs["supervised"] = axis_names

    f[f"{iteration}/supervised_labels"] = (
        batch["supervised_labels"].detach().cpu().numpy()
    )
    f[f"{iteration}/supervised_labels"].attrs["supervised"] = axis_names

    f[f"{iteration}/prediction"] = prediction.detach().cpu().numpy()
    f[f"{iteration}/prediction"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"].attrs["offset"] = prediction_offset

    print(f"Snapshot saved at iteration {iteration}")
