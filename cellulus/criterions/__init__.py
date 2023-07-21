from cellulus.criterions.oce_loss import OCELoss


def get_loss(
    temperature, regularizer_weight, density, kappa, num_spatial_dims, reduce_mean
):
    return OCELoss(
        temperature, regularizer_weight, density, kappa, num_spatial_dims, reduce_mean
    )
