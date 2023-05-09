import pytorch_lightning as pl
from torch.utils.data import DataLoader
from VerySimpleSSLTrainer import VerySimpleSSLTrainer
from colocseg.datasets import  ZarrImageDataset
from datasets.SimplifiedCoordinateDataset import SimplifiedCoordinateDataset

train_ds_zarr = ZarrImageDataset(r"C:\Users\User\Documents\lightsheet-model\data\lightsheet\lightsheet_cagtag_200_0_1.zarr",
                                    "train",
                                    crop_to = (128,128,128))

train_ds_coord = SimplifiedCoordinateDataset(train_ds_zarr)

val_ds_zarr = ZarrImageDataset(r"C:\Users\User\Documents\lightsheet-model\data\lightsheet\lightsheet_cagtag_200_3_0.zarr",
                                    "train",
                                    crop_to = (128,128,128))

val_ds_coord = SimplifiedCoordinateDataset(val_ds_zarr)

loader = DataLoader(train_ds_coord,num_workers=0)
val_loader = DataLoader(val_ds_coord,num_workers=0)

trainer = pl.Trainer(limit_train_batches=50, max_epochs=1, accelerator='gpu', devices=1, fast_dev_run=False)
model = VerySimpleSSLTrainer()

trainer.fit(model = model, train_dataloaders = loader, val_dataloaders=val_loader)
print("Fitting Finished")