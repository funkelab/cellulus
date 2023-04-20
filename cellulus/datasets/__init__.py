from cellulus.datasets.TwoDimensionalDataset import TwoDimensionalDataset
def dataset(name, dataset_opts):
    if name =="2d":
        return TwoDimensionalDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))