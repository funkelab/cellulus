from cellulus.criterions.oce_loss import OCELoss


def get_loss(temperature, regularizer_weight, density, kappa):
    return OCELoss(temperature, regularizer_weight, density, kappa)
