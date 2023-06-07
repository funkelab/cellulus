from cellulus.datasets.dataset_2D import Dataset2D


def get_dataset(name, dataset_opts):
    if name == "2D":
        return Dataset2D(**dataset_opts)
    else:
        raise RuntimeError(f"Dataset {name} not available")
