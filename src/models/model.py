import logging

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AdamW, ConvBertForSequenceClassification

from src.utils import all_logging_disabled


class ConvBert(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConvBert")
        parser.add_argument("--lr", default=0.001)
        parser.add_argument("--fine_tune_layers", default=1, type=int)
        parser.add_argument("--last_layer_dropout", default=0.1, type=float)
        return parent_parser

    def __init__(self, lr, fine_tune_layers, last_layer_dropout, **kwargs):

        super().__init__()

        self.save_hyperparameters(
            "lr", "fine_tune_layers", "last_layer_dropout"
        )

        self.lr = lr
        self.fine_tune_layers = fine_tune_layers
        self.last_layer_dropout = last_layer_dropout

        with all_logging_disabled(logging.ERROR):
            model = ConvBertForSequenceClassification.from_pretrained(
                "YituTech/conv-bert-base"
            )
        self.model = model

        self.model._modules["classifier"].dropout.p = last_layer_dropout

        for param in self.model._modules["convbert"].embeddings.parameters():
            param.requires_grad = False

        if fine_tune_layers == 0:
            layer_slice = slice(None)
        else:
            layer_slice = slice(None, -fine_tune_layers)

        params_to_freeze = (
            self.model._modules["convbert"]
            .encoder.layer[layer_slice]
            .parameters()
        )
        for param in params_to_freeze:
            param.requires_grad = False

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3].unsqueeze(1),
        }

        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        self.log("train_loss", output.loss)

        return output.loss

    def validation_step(self, batch, batch_idx):
        batch = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3].unsqueeze(1),
        }

        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        predictions = output.logits.argmax(-1)

        self.log(
            "val_accuracy",
            self.accuracy(predictions, batch["labels"].squeeze()),
            prog_bar=True,
        )
        self.log("val_loss", output.loss)

    def test_step(self, batch, batch_dix):

        batch = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
        }

        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )

        probs = output.logits.softmax(-1)

        return probs

    def test_epoch_end(self, test_step_outputs):

        probs = torch.cat(test_step_outputs)

        test_data_raw = pd.read_csv("./data/raw/test.csv")

        test_submission = pd.DataFrame(
            {"id": test_data_raw["id"], "target": probs.argmax(-1)}
        )

        test_submission.to_csv("results/kaggle_submission.csv", index=False)

        # wandb.init(
        #     project="mlops_project",
        #     entity="mlops_project",
        #     group="test-samples",
        # )

        # test_samples = pd.DataFrame({
        #     "id": test_data_raw["id"][:32],
        #     "tweet" :  test_data_raw["text"][:32],
        #     "p_real_disaster": probs[:32, 1],
        #     "p_not_real_disaster": probs[:32, 0]
        #     })

        # wandb.log({"test_samples" : wandb.Table(dataframe = test_samples)})

        # wandb.finish()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        return optimizer


if __name__ == "__main__":

    model = ConvBert(0.001, 5, 0.5)
