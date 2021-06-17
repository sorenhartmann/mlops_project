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
        parser.add_argument('--fine_tune_layers', default=1, type=int)
        return parent_parser

    def __init__(self, lr, fine_tune_layers, **kwargs):

        super().__init__()

        self.save_hyperparameters("lr", "fine_tune_layers")

        self.lr = lr
        self.fine_tune_layers = fine_tune_layers

        with all_logging_disabled(logging.ERROR):
            model = ConvBertForSequenceClassification.from_pretrained('YituTech/conv-bert-base')
        self.model = model

        for param in self.model._modules['convbert'].embeddings.parameters():
            param.requires_grad = False

        if fine_tune_layers == 0:
            layer_slice = slice(None)
        else:
            layer_slice = slice(None, -fine_tune_layers)

        params_to_freeze = (
            self
            .model
            ._modules['convbert']
            .encoder
            .layer[layer_slice]
            .parameters()
        )
        for param in params_to_freeze:
            param.requires_grad = False


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
        optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                )

        return optimizer