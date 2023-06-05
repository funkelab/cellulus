from cellulus.datasets.TwoDimensionalDataset import TwoDimensionalDataset
def get_dataset(name, dataset_opts):
    if name =="2D":
        return TwoDimensionalDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))