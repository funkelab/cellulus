import os
import torch
import cellulus.utils.transforms as my_transforms


def create_dataset_dict(
    data_dir,
    project_name,
    type,
    crop_size,
    norm="min-max-percentile",
    density=0.1,  # TODO: in paper, we say `0.2`
    radius=32,  # TODO: in paper, we say `10` pixels
    name="2D",
    batch_size=8,
    workers=8,
):
    """
    Creates `dataset_dict` dictionary from parameters.
    Parameters
    ----------
    data_dir: string
        Data shall be read from os.path.join(data_dir, project_name)
    project_name: string
        Data shall be read from os.path.join(data_dir, project_name)
    type: string
        One of 'train', 'val'
    name: string
        One of '2D' or '3D'
    batch_size: int

    workers: int
        Number of data-loader workers
    """
    if name == "2D":
        set_transforms = my_transforms.get_transform(
            [
                # TODO --> include rotations and flips
                # But what happens to the sampled coordinates?
                # They should be rotated accordingly ...
                # {
                #     "name": "RandomRotationsAndFlips",
                #     "opts": {
                #         "keys": ("image",),
                #         "degrees": 90,
                #     },
                # },
                {
                    "name": "ToTensorFromNumpy",
                    "opts": {
                        "keys": ("image",),
                        "type": (torch.FloatTensor),
                    },
                },
            ]
        )
    dataset_dict = {
        "name": name,
        "kwargs": {
            "data_dir": os.path.join(data_dir, project_name),
            "type": type,
            "transform": set_transforms,
            "crop_size": crop_size,
            "norm": norm,
            "density": density,
            "radius": radius,
        },
        "batch_size": batch_size,
        "workers": workers,
    }
    print(
        "`{}_dataset_dict` dictionary successfully created with: \n "
        "-- {} images accessed from {}, "
        "\n -- batch size set at {}, "
        "\n -- crop size set at {}".format(
            type,
            type,
            os.path.join(data_dir, project_name, type + ".zarr"),
            batch_size,
            crop_size,
        )
    )
    return dataset_dict


def create_model_dict(
    num_input_channels, num_output_channels=2, name="2D", num_fmaps=256
):
    """
    Creates `model_dict` dictionary from parameters.
    Parameters
    ----------
    num_input_channels: int
        1 indicates gray-channel image
    num_output_channels: int
        2 indicates the two offsets predicted along y and x directions
    name: string
    """
    model_dict = {
        "name": "UNet2D" if name == "2D" else "UNet3D",
        "kwargs": {
            "out_channels": num_output_channels,
            "in_channels": num_input_channels,
            "num_fmaps": num_fmaps,
        },
    }
    print(
        "`model_dict` dictionary successfully created with: "
        "\n -- num of output channels equal to {}, "
        "\n -- num of input channels equal to {}, "
        "\n -- name equal to {}".format(
            num_output_channels, num_input_channels, model_dict["name"]
        )
    )
    return model_dict


def create_loss_dict(temperature=10.0, regularization_weight=1e-5):
    """
    Creates `loss_dict` dictionary from parameters.
    Parameters
    ----------
    temperature: float

    regularization_weight: float


    """
    loss_dict = {
        "lossOpts": {
            "temperature": temperature,
            "regularization_weight": regularization_weight,
        }
    }
    print(
        "`loss_dict` dictionary successfully created with: "
        "\n -- regularization weight equal to {:.3f} and temperature equal to {:.3f}".format(
            regularization_weight, temperature
        )
    )
    return loss_dict


def create_configs(
    save_dir,
    resume_path,
    n_epochs=50,
    train_lr=5e-4,
    cuda=True,
    save=True,
    save_checkpoint_frequency=None,
):
    """
    Creates `configs` dictionary from parameters.
    Parameters
    ----------
    save_dir: str
        Path to where the experiment is saved
    resume_path: str
        Path to where the trained model (for example, checkpoint.pth) lives
    n_epochs: int
        Total number of epochs
    train_lr: float
        Starting learning rate
    cuda: boolean
        If True, use GPU
    save: boolean
        If True, then results are saved
    save_checkpoint_frequency: int
        Save model weights after 'n' epochs (in addition to last and best model weights)
        Default is None
    """
    configs = dict(
        train_lr=train_lr,
        n_epochs=n_epochs,
        cuda=cuda,
        save=save,
        save_dir=save_dir,
        resume_path=resume_path,
        save_checkpoint_frequency=save_checkpoint_frequency,
    )
    print(
        "`configs` dictionary successfully created with: "
        "\n -- n_epochs equal to {}, "
        "\n -- save_dir equal to {}, "
        "\n -- cuda set to {}".format(n_epochs, save_dir, cuda)
    )
    return configs
