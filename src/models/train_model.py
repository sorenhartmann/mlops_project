from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import PyTorchProfiler

from src.data.datamodule import DisasterDataModule
from src.models.model import ConvBert


def main(args):

    dm = DisasterDataModule("./data", batch_size=16)
    model = ConvBert(**vars(args))

    wandb_logger = WandbLogger(project="mlops_project", entity="mlops_project")

    if args.Aprofiler:
        # profiler =
        # AdvancedProfiler(dirpath = 'reports', filename = 'profiler')
        profiler = PyTorchProfiler(
            dirpath="reports",
            filename="profiler.prof",
            sort_by_key="cuda_time",
        )
    else:
        profiler = None

    trainer = pl.Trainer.from_argparse_args(
        args, profiler=profiler, logger=wandb_logger, accelerator="ddp"
    )

    trainer.fit(model, dm)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser = ConvBert.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--Aprofiler", action="store_true")

    args = parser.parse_args()

    main(args)
