import sys
sys.path.append("C:/Users/User/Documents/lightsheet-model/train-lightsheet-model/")


from argparse import ArgumentParser
from colocseg.datamodules import TissueNetDataModule
# from datamodules import TissueNetDataModule
# from colocseg.evaluation import AnchorSegmentationValidation
from evaluation import AnchorSegmentationValidation
# from colocseg.trainingmodules import SSLTrainer
from trainingmodules import SSLTrainer
# from colocseg.utils import SaveModelOnValidation
from utils import SaveModelOnValidation
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl

def main():
    model = SSLTrainer()

    trainer = pl.Trainer(limit_train_batches=1, max_epochs=1, devices=1, accelerator="gpu")

    datamodule = TissueNetDataModule(batch_size=10,dspath="C:\\Users\\User\\Downloads\\tissuenet_v1.0\\tissuenet_1.0",positive_radius=[0,32])

    model = SSLTrainer()

    anchor_val = AnchorSegmentationValidation(run_segmentation=False)
    lr_logger = LearningRateMonitor(logging_interval='step')
    model_saver = SaveModelOnValidation()


    trainer.callbacks.append(model_saver)
    trainer.callbacks.append(anchor_val)
    trainer.callbacks.append(lr_logger)

    trainer.fit(model = model, datamodule = datamodule)
    print("Fitting Finished")




if __name__ == '__main__':
    main()     