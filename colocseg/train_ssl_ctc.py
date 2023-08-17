from argparse import ArgumentParser
from colocseg.datamodules import CTCDataModule
from colocseg.evaluation import AnchorSegmentationValidation
from colocseg.trainingmodules import SSLTrainer
from colocseg.utils import SaveModelOnValidation
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl


if __name__ == '__main__':

    parser = ArgumentParser()

    pl.utilities.seed.seed_everything(42)
    parser = SSLTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)  # , logger=WandbLogger(project='SSLAnchor'))
    parser = CTCDataModule.add_argparse_args(parser)
    parser = CTCDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # init module
    model = SSLTrainer.from_argparse_args(args)

    datamodule = CTCDataModule.from_argparse_args(args)
    anchor_val = AnchorSegmentationValidation(run_segmentation=False)
    lr_logger = LearningRateMonitor(logging_interval='step')
    model_saver = SaveModelOnValidation()

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.callbacks.append(model_saver)
    trainer.callbacks.append(anchor_val)
    trainer.callbacks.append(lr_logger)
    trainer.fit(model, datamodule)
