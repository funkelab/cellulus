import torch
import zarr
from tqdm import tqdm

from cellulus.criterions import get_loss
from cellulus.datasets import get_dataset
from cellulus.models import get_model

torch.backends.cudnn.benchmark = True


def train(experiment_config):
    print(experiment_config)

    # create train dataset
    train_dataset = get_dataset(
        dataset_config=experiment_config.train_config.train_data_config,
        crop_size=experiment_config.train_config.crop_size,
    )

    # create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=experiment_config.train_config.batch_size,
        drop_last=True,
        num_workers=experiment_config.train_config.num_workers,
        pin_memory=True,
    )

    # set model
    model = get_model(
        in_channels=train_dataset.get_num_channels(),
        out_channels=train_dataset.get_num_spatial_dims(),
        num_fmaps=experiment_config.model_config.num_fmaps,
        fmap_inc_factor=experiment_config.model_config.fmap_inc_factor,
        features_in_last_layer=experiment_config.model_config.features_in_last_layer,
        downsampling_factors=experiment_config.model_config.downsampling_factors,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
    )

    model = model.cuda()

    # set loss
    criterion = get_loss(
        regularizer_weight=experiment_config.train_config.regularizer_weight,
        temperature=experiment_config.train_config.temperature,
        kappa=experiment_config.train_config.kappa,
        density=experiment_config.train_config.density,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_config.train_config.initial_learning_rate,
    )

    # set scheduler:

    # resume training
    start_iteration = 0
    if experiment_config.model_config.checkpoint is None:
        pass
    else:
        print(f"Resuming model from {experiment_config.checkpoint}")
        state = torch.load(experiment_config.model_config.checkpoint)
        start_iteration = state["iteration"] + 1
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])

    # call `train_iteration`
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, experiment_config.train_config.max_iterations),
            train_dataloader,
        )
    ):
        train_loss, prediction = train_iteration(
            batch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        print(f"===> train loss: {train_loss:.2f}")

        state = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        if iteration % experiment_config.train_config.save_model_every == 0:
            save_model(
                state,
                iteration,
            )
        if iteration % experiment_config.train_config.save_snapshot_every == 0:
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
    pass


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
