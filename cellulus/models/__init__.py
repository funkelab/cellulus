from cellulus.models.unet_2D import UNet2D


def get_model(name, model_opts):
    if name == "UNet2D":
        return UNet2D(**model_opts)
    else:
        raise RuntimeError(f"Model {name} not available")
