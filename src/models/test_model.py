from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import wandb

from src.data.datamodule import DisasterDataModule
from src.models.model import ConvBert

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model_name = args.model_name

    wandb_entity = "mlops_project"
    wandb_project = "mlops_project"

    # Get runs from wandb
    wandb_api = wandb.Api()
    runs = wandb_api.runs(f"{wandb_entity}/{wandb_project}")

    # Get relevant run
    run = next(run for run in runs if run.name == model_name)

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
    model.eval()

    dm = DisasterDataModule("./data", batch_size=32)

    trainer = pl.Trainer.from_argparse_args(
        args,
        limit_train_batches=0,
        limit_val_batches=0,
    )

    trainer.test(model, datamodule=dm)

    # input_tweet = request.args.get("string")

    # app.logger.info(f"Received string: '{input_tweet}'")

    # app.logger.info("Loading tokenizer")
    # tokenizer = load_tokenizer()
    # app.logger.info("Loaded tokenizer!")
    # app.logger.info("Loading model")
    # model = load_model(model_name)
    # app.logger.info("Loaded model!")

    # input_ids, attention_mask, token_type_ids = tokenizer.tokenize(
    #     [input_tweet]
    # )
    # app.logger.info("Forward pass...")
    # output = model(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     token_type_ids=token_type_ids,
    # )

    # probs = output.logits.softmax(-1).squeeze()

    # app.logger.info(
    #     f"P( Real disaster ) = {probs[1]:.2f}% / "
    #     f"P( not real disaster ) = {probs[0]:.2f}%"
    # )

    # return jsonify(
    #     p_real_disaster=probs[1].item(),
    #     p_not_real_disaster=probs[0].item(),
    # )
