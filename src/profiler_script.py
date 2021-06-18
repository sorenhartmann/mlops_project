import argparse
import cProfile
import os
import pstats
import subprocess
import sys
from pstats import SortKey

from pytorch_lightning import Trainer

from src.models.main_pl import MNIST_model, MNISTDataModule, PrintCallback

parser = argparse.ArgumentParser(description="Profiling arguments")
parser.add_argument("--gpus", default=0)
parser.add_argument("--accelerator", default="ddp")
parser.add_argument("--epochs", default=1)
parser.add_argument("--train_size", default=0.2)
parser.add_argument("--version", default=0)
parser.add_argument("--prints", default=10)
parser.add_argument("--visualize", action="store_true")

args = parser.parse_args(sys.argv[1:2])


def profiler_(trainer, model, data, visualize, version, prints):
    if not os.path.exists("profiles"):
        os.makedirs("profiles")

    profile_path = "profiles/training_stats_" + str(version) + ".prof"
    cProfile.run("trainer.fit(model, data)", profile_path)
    p = pstats.Stats(profile_path)

    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(int(prints))

    if visualize:
        subprocess.call(["snakeviz", profile_path])


if __name__ == "__main__":
    model = MNIST_model()
    data = MNISTDataModule()

    callbacks = [PrintCallback()]
    trainer = Trainer(
        max_epochs=int(args.epochs),
        limit_train_batches=args.train_size,
        accelerator=args.accelerator,
        gpus=args.gpus,
        callbacks=callbacks,
    )
    profiler_(trainer, model, data, args.visualize, args.version, args.prints)
