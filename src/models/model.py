from transformers import ConvBertForSequenceClassification, AdamW
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
import argparse
from src.utils import all_logging_disabled
import logging
import functools
import torchmetrics

class ConvBert(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConvBert")
        parser.add_argument('--lr', default=0.001)
        return parent_parser

    def __init__(self, lr, **kwargs):

        super().__init__()

        self.save_hyperparameters("lr")

        self.lr = lr

        with all_logging_disabled(logging.ERROR):
            model = ConvBertForSequenceClassification.from_pretrained('YituTech/conv-bert-base')
        self.model = model

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3].unsqueeze(1)
            }

        output = self(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )

        self.log("train_loss", output.loss)

        return output.loss


    def validation_step(self, batch, batch_idx):
        batch = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3].unsqueeze(1)
            }

        output = self(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )

        predictions = output.logits.argmax(-1)
        self.log("val_accuracy", self.accuracy(predictions, batch["labels"].squeeze()), prog_bar=True)
        self.log('val_loss', output.loss)


    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.lr,
                )

        return optimizer