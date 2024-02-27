from cellulus.criterions.oce_loss import OCELoss


def get_loss(
    temperature,
    regularizer_weight,
    density,
    num_spatial_dims,
    device,
):
    return OCELoss(
        temperature,
        regularizer_weight,
        density,
        num_spatial_dims,
        device,
    )
