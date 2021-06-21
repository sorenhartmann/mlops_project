import argparse
import os
import sys
from pathlib import Path

import wandb

from src.models.model import ConvBert

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model deploying arguments")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version", default=1.0)
    parser.add_argument("--serialize_model", action="store_true")
    parser.add_argument("--deploy_model", action="store_true")
    parser.add_argument("--model_path", default="models/checkpoint.pth")
    args = parser.parse_args(sys.argv[1:2])

    wandb_entity = "mlops_project"
    wandb_project = "mlops_project"

    # Get runs from wandb
    wandb_api = wandb.Api()
    runs = wandb_api.runs(f"{wandb_entity}/{wandb_project}")

    # Get relevant run
    if args.model_name is None:
        run = next(run for run in runs if run.state == "finished")
    else:
        run = next(run for run in runs if run.name == args.model_name)

    # Find checkpoint file, possibly downloading into ./models/
    checkpoint_file = next(
        file for file in run.files() if file.name.endswith(".ckpt")
    )

    try:
        local_path = next(
            Path("./wandb").glob(f"*-{run.id}/files/{checkpoint_file.name}")
        )
    except StopIteration:
        local_path = Path("models") / checkpoint_file.name
        if not local_path.exists():
            checkpoint_file.download("models").close()

    model = ConvBert.load_from_checkpoint(local_path)
    # model = ConvBert(lr=0.01, **{})

    # # #TODO insert load of state dict

    # script_model = torch.jit.script(model)
    # script_model.save('torch_serve/' + str(args.model_name) + '.pt')
    os.makedirs("model_store/", exist_ok=True)
    if args.serialize_model:
        ss = (
            "torch-model-archiver --model-name "
            + str(args.model_name)
            + " --version "
            + str(args.version)
            + " --serialized-file "
        )
        ss = (
            ss
            + "torch_serve/"
            + str(args.model_name)
            + ".pt"
            + " --export-path model_store"
            + " --extra-files torch_serve/index_to_name.json"
            + " --handler text_classifier"
        )
        os.system(ss)

    if args.deploy_model:
        ss = (
            "torchserve --start --ncs --model-store model_store --models "
            + str(args.model_name)
            + "="
            + str(args.model_name)
            + ".mar"
        )
        os.system(ss)
