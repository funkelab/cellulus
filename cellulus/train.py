import os

import torch
import zarr
from tqdm import tqdm

from cellulus.criterions import get_loss
from cellulus.datasets import get_dataset
from cellulus.models import get_model
from cellulus.utils import get_logger

torch.backends.cudnn.benchmark = True


def train(experiment_config):
    print(experiment_config)

    if not os.path.exists("models"):
        os.makedirs("models")

    train_config = experiment_config.train_config
    model_config = experiment_config.model_config

    # create train dataset
    train_dataset = get_dataset(
        mode="train",
        dataset_config=train_config.train_data_config,
        crop_size=train_config.crop_size,
        control_point_spacing=train_config.control_point_spacing,
        control_point_jitter=train_config.control_point_jitter,
    )

    # create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    # set model
    model = get_model(
        in_channels=train_dataset.get_num_channels(),
        out_channels=train_dataset.get_num_spatial_dims(),
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=model_config.downsampling_factors,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
    )

    model = model.cuda()

    # set loss
    criterion = get_loss(
        regularizer_weight=train_config.regularizer_weight,
        temperature=train_config.temperature,
        kappa=train_config.kappa,
        density=train_config.density,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.initial_learning_rate,
    )

    # set scheduler:

    # set logger
    logger = get_logger(keys=["train"], title="loss")

    # resume training
    start_iteration = 0
    if model_config.checkpoint is None:
        pass
    else:
        print(f"Resuming model from {model_config.checkpoint}")
        state = torch.load(model_config.checkpoint)
        start_iteration = state["iteration"] + 1
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])

    # call `train_iteration`
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, train_config.max_iterations),
            train_dataloader,
        )
    ):
        train_loss, prediction = train_iteration(
            batch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        print(f"===> train loss: {train_loss:.6f}")
        logger.add(key="train", value=train_loss)
        logger.write()
        logger.plot()
        state = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "logger_data": logger.data,
        }
        if iteration % train_config.save_model_every == 0:
            save_model(
                state,
                iteration,
            )
        if iteration % train_config.save_snapshot_every == 0:
            save_snapshot(
                batch,
                prediction,
                iteration,
            )


def train_iteration(
    batch,
    model,
    criterion,
    optimizer,
):
    model.train()
    prediction = model(batch.cuda())
    loss = criterion(prediction)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), prediction


def save_model(state, iteration):
    file_name = os.path.join("models", str(iteration).zfill(6) + ".pth")
    torch.save(state, file_name)
    print(f"Checkpoint saved at iteration {iteration}")


def save_snapshot(batch, prediction, iteration):
    num_spatial_dims = len(batch.shape) - 2

    axis_names = ["s", "c"] + ["t", "z", "y", "x"][-num_spatial_dims:]
    prediction_offset = tuple(
        (a - b) / 2
        for a, b in zip(
            batch.shape[-num_spatial_dims:], prediction.shape[-num_spatial_dims:]
        )
    )
    f = zarr.open("snapshots.zarr", "a")
    f[f"{iteration}/raw"] = batch.detach().cpu().numpy()
    f[f"{iteration}/raw"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"] = prediction.detach().cpu().numpy()
    f[f"{iteration}/prediction"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"].attrs["offset"] = prediction_offset

    print(f"Snapshot saved at iteration {iteration}")
