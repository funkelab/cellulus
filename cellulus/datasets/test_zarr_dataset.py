from cellulus.datasets.zarr_dataset import ZarrDataset
from cellulus.configs import DatasetConfig

class TestZarrDataset(ZarrDataset): # type : ignore
    def __init__(self, dataset_config, crop_size):
        self.dataset_config = dataset_config
        self.crop_size = crop_size
        ZarrDataset.read_meta_data(self)
        self.__setup_pipeline()

    def __setup_pipeline(self):
        pass


