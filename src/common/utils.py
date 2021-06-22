import os
from pathlib import Path
from typing import Optional

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def get_env(env_name: str, default: Optional[str] = None) -> str:

    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!"
            )
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured"
                "and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:

    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:

    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(
        p.numel() for p in model.parameters()
    )
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging
    # hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
