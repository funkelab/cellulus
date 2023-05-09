import pytorch_lightning as pl
from datasets.Idealised_Image_Datasets import Idealised3DImageDataset
from torch.utils.data import DataLoader
from VerySimpleSSLTrainer import VerySimpleSSLTrainer

dataset = Idealised3DImageDataset(batch_size=10240000,img_size=128,num_objects=32,object_radius=7,radius=5)
idealised_loader = DataLoader(dataset,num_workers=0)
model = VerySimpleSSLTrainer()

trainer = pl.Trainer(limit_train_batches=25, max_epochs=1, accelerator='gpu',fast_dev_run=True)
trainer.fit(model=model, train_dataloaders = idealised_loader)
print("Fitting Finished")