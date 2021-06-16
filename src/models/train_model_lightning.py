from transformers import ConvBertForSequenceClassification
from src.data.build_loader import build_train_val_loader, build_test_loader, terminal_colors
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from src.data.datamodule import DisasterDataModule


class ConvBert(pl.LightningModule):

    def __init__(self):
        super(ConvBert, self).__init__()

        #parser = argparse.ArgumentParser(description='Training arguments')
        #parser.add_argument('--lr', default=0.003)
        #parser.add_argument('--epoch', default = 10)
        #parser.add_argument('--batch_size', default=16)
        #parser.add_argument('--val_size', default=0.2)
        #parser.add_argument('--model_version', default = 0)
        #args = parser.parse_args(sys.argv[2:])

        model = ConvBertForSequenceClassification.from_pretrained('YituTech/conv-bert-base')
        self.model = model

    
    def training_step(self, batch, batch_idx):
        batch = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3].unsqueeze(1)
            }

        loss, _ = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3].unsqueeze(1)
            }

        val_loss, _ = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )
        self.log('val_loss', val_loss)


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
        optimizer = Adam(
                optimizer_grouped_parameters,
                lr=0.003,
                )
        return optimizer




if __name__ == '__main__':
    dm = DisasterDataModule("./data", batch_size=16)
    dm.prepare_data()

    dm.setup()

    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()

    
    trainer = pl.Trainer()

    model = ConvBert()

    trainer.fit(model, trainloader)
