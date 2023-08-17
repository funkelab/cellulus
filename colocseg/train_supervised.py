from argparse import ArgumentParser
from colocseg.datamodules import PartiallySupervisedDataModule
from colocseg.evaluation import StardistSegmentationValidation, CellposeSegmentationValidation
from colocseg.trainingmodules import PartiallySupervisedTrainer
from colocseg.utils import SaveModelOnValidation
from colocseg.test_inference import test_from_checkpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl


if __name__ == '__main__':

    parser = ArgumentParser()

    # pl.utilities.seed.seed_everything(42)
    parser = PartiallySupervisedTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PartiallySupervisedDataModule.add_argparse_args(parser)
    parser = PartiallySupervisedDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # init module
    model = PartiallySupervisedTrainer.from_argparse_args(args)
    datamodule = PartiallySupervisedDataModule.from_argparse_args(args)
    
    if args.loss_name_super == 'StardistLoss':
        seg_val = StardistSegmentationValidation(f"Colocseg_TissueNet_{args.limit}", min_size=0)
    elif args.loss_name_super == 'CellposeLoss':
        seg_val = CellposeSegmentationValidation(f"Colocseg_TissueNet_{args.limit}", min_size=0)
    else:
        raise NotImplementedError("Validator for loss not implemented")
    
    lr_logger = LearningRateMonitor(logging_interval='step')
    model_saver = SaveModelOnValidation()

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.callbacks.append(model_saver)
    trainer.callbacks.append(seg_val)
    trainer.callbacks.append(lr_logger)
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)

    # compute test scores
    if args.tissue_type == 'all':
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type='all')
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type="immune")
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type="pancreas")
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type="lung")
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type="skin")
    else:
        test_from_checkpoint(".", args, checkpoint_index=0, target_type=args.target_type, tissue_type=args.tissue_type)
