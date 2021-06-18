from argparse import ArgumentParser

import optuna
import pytorch_lightning as pl
import wandb

from src.data.datamodule import DisasterDataModule
from src.models.model import ConvBert


class Objective:
    def __init__(self, args) -> None:
        self.args = args

    def __call__(self, trial):

        wandb_logger = pl.loggers.WandbLogger(
            project="mlops_project",
            entity="mlops_project",
            tags=["hparam-search"],
            group="hparam-search",
        )

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        fine_tune_layers = trial.suggest_int("fine_tune_layers", 0, 5)

        wandb_logger.log_hyperparams({"batch_size": batch_size})
        dm = DisasterDataModule(batch_size=batch_size, data_dir="./data")
        model = ConvBert(lr=lr, fine_tune_layers=fine_tune_layers)

        trainer = pl.Trainer.from_argparse_args(
            self.args, logger=wandb_logger, checkpoint_callback=False
        )
        trainer.fit(model, dm)

        wandb.finish()

        return trainer.callback_metrics["val_accuracy"].item()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    objective = Objective(args)

    storage_name = "sqlite:///optuna-storage.db"

    study = optuna.create_study(
        study_name="hparam-search",
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, timeout=1800)
