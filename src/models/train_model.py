from src.data.build_loader import build_train_val_loader, build_test_loader, terminal_colors
import torch
import torch.nn as nn
import sys
import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from src.data.datamodule import DisasterDataModule
from src.models.model import ConvBert
import wandb
from pytorch_lightning.profiler import AdvancedProfiler,PyTorchProfiler
def main(args):

    wandb_api = wandb.Api()

    wandb_logger = WandbLogger(project="mlops_project", entity="mlops_project")
    
    dm = DisasterDataModule("./data", batch_size=16)

    model = ConvBert(**vars(args))

    if args.Aprofiler:
        #profiler = AdvancedProfiler(dirpath = 'reports', filename = 'profiler')
        profiler = PyTorchProfiler(dirpath = 'reports', filename = 'profiler.prof', sort_by_key = 'cuda_time')
    else:
        profiler = None

    trainer = pl.Trainer.from_argparse_args(
        args,
        profiler = profiler,
        logger=wandb_logger,
        accelerator ='ddp'
    )

    trainer.fit(model, dm)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser = ConvBert.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--Aprofiler', action='store_true')

    args = parser.parse_args()
    

    main(args)