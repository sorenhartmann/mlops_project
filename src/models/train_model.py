from transformers import ConvBertForSequenceClassification
from src.data.build_loader import build_train_val_loader, build_test_loader
import torch
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=0.003)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--val_size', default=0.2)
args = parser.parse_args(sys.argv[2:])


def train():
    #Import dataset and create dataloaders
    trainset = pd.read_csv('data/preprocessed/train.csv')
    testset = pd.read_csv('data/preprocessed/test.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, valloader = build_train_val_loader(trainset, args.batch_size,
                                                    device, args.val_size)
    testloader = build_test_loader(testset, args.batch_size, device)

    #Import model:
    model = ConvBertForSequenceClassification.from_pretrained(
        'YituTech/conv-bert-base')

    batch = next(iter(trainloader))
    batch = {
        'input_ids': batch[0],
        'token_type_ids': batch[1],
        'attention_mask': batch[2],
        'labels': batch[3]
    }
    print(model(**batch))


if __name__ == '__main__':
    train()
