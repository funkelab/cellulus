import os

import numpy as np
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
        dataset_config=train_config.train_data_config,
        crop_size=tuple(train_config.crop_size),
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
        downsampling_factors=[
            tuple(factor) for factor in model_config.downsampling_factors
        ],
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
    )

    # set device
    device = torch.device(train_config.device)

    model = model.to(device)

    # set loss
    criterion = get_loss(
        regularizer_weight=train_config.regularizer_weight,
        temperature=train_config.temperature,
        kappa=train_config.kappa,
        density=train_config.density,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
        reduce_mean=train_config.reduce_mean,
        device=device,
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.initial_learning_rate,
    )

    # set scheduler:

    def lambda_(iteration):
        return pow((1 - ((iteration) / train_config.max_iterations)), 0.9)

    # set logger
    logger = get_logger(keys=["train"], title="loss")

    # resume training
    start_iteration = 0
    lowest_loss = 1e0

    if model_config.checkpoint is None:
        pass
    else:
        print(f"Resuming model from {model_config.checkpoint}")
        state = torch.load(model_config.checkpoint, map_location=device)
        start_iteration = state["iteration"] + 1
        lowest_loss = state["lowest_loss"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    # call `train_iteration`
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, train_config.max_iterations),
            train_dataloader,
        )
    ):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
        )

        train_loss, prediction = train_iteration(
            batch, model=model, criterion=criterion, optimizer=optimizer, device=device
        )
        scheduler.step()
        print(f"===> train loss: {train_loss:.6f}")
        logger.add(key="train", value=train_loss)
        logger.write()
        logger.plot()

        if iteration % train_config.save_model_every == 0:
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

        if iteration % train_config.save_snapshot_every == 0:
            save_snapshot(
                batch,
                prediction,
                iteration,
            )


def train_iteration(batch, model, criterion, optimizer, device):
    model.train()
    prediction = model(batch.to(device))
    loss = criterion(prediction)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), prediction


def save_model(state, iteration, is_lowest=False):
    file_name = os.path.join("models", str(iteration).zfill(6) + ".pth")
    torch.save(state, file_name)
    print(f"Checkpoint saved at iteration {iteration}")
    if is_lowest:
        file_name = os.path.join("models", "best_loss.pth")
        torch.save(state, file_name)


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
    f[f"{iteration}/raw"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims

    # normalize the offsets by subtracting the mean offset per image
    prediction_cpu = prediction.detach().cpu().numpy()
    prediction_cpu_reshaped = np.reshape(
        prediction_cpu, (prediction_cpu.shape[0], prediction_cpu.shape[1], -1)
    )
    mean_prediction = np.mean(prediction_cpu_reshaped, 2)
    prediction_cpu -= mean_prediction[(...,) + (np.newaxis,) * num_spatial_dims]
    f[f"{iteration}/prediction"] = prediction_cpu
    f[f"{iteration}/prediction"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"].attrs["offset"] = prediction_offset
    f[f"{iteration}/prediction"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims

    print(f"Snapshot saved at iteration {iteration}")
