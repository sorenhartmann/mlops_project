import logging
from pathlib import Path

import wandb
from flask import Flask, request
from flask.json import jsonify
from flask_caching import Cache

from src.data.preprocessing import Tokenizer
from src.models.model import ConvBert


def build_app(model_name=None):

    cache = Cache(config={"CACHE_TYPE": "SimpleCache"})
    app = Flask(__name__)
    cache.init_app(app)

    @cache.memoize()
    def load_tokenizer():

        return Tokenizer()

    @cache.memoize()
    def load_model(model_name):

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
        app.logger.info(
            f"Found model with checkpoint path: '{checkpoint_file.name}'"
        )

        try:
            local_path = next(
                Path("./wandb").glob(
                    f"*-{run.id}/files/{checkpoint_file.name}"
                )
            )
            app.logger.info("Found model checkpoint in local wandb storage")
        except StopIteration:
            local_path = Path("models") / checkpoint_file.name
            if not local_path.exists():
                app.logger.info("Downloading model checkpoint to ./models/")
                checkpoint_file.download("models").close()
                app.logger.info("Download finished!")
            else:
                app.logger.info("Found model checkpoint in ./models/")

        model = ConvBert.load_from_checkpoint(local_path)
        model.eval()

        return model

    @app.route("/predict", methods=["GET"])
    def hello_word():

        input_tweet = request.args.get("string")

        app.logger.info(f"Received string: '{input_tweet}'")

        app.logger.info("Loading tokenizer")
        tokenizer = load_tokenizer()
        app.logger.info("Loaded tokenizer!")
        app.logger.info("Loading model")
        model = load_model(model_name)
        app.logger.info("Loaded model!")

        input_ids, attention_mask, token_type_ids = tokenizer.tokenize(
            [input_tweet]
        )
        app.logger.info("Forward pass...")
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        probs = output.logits.softmax(-1).squeeze()

        app.logger.info(
            f"P( Real disaster ) = {probs[1]:.2f}% / "
            f"P( not real disaster ) = {probs[0]:.2f}%"
        )

        return jsonify(
            p_real_disaster=probs[1].item(),
            p_not_real_disaster=probs[0].item(),
        )

    app.logger.setLevel(logging.INFO)

    return app
