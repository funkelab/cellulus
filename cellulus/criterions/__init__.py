from cellulus.criterions.my_loss import OCELoss
def get_loss(loss_opts):
    return OCELoss(**loss_opts)
