from transformers import ConvBertForSequenceClassification
from src.data.build_loader import build_train_val_loader, build_test_loader, terminal_colors
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
from src.data.datamodule import DisasterDataModule

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=0.003)
parser.add_argument('--epoch', default = 10)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--val_size', default=0.2)
parser.add_argument('--model_version', default = 0)
args = parser.parse_args(sys.argv[2:])

def train_func(trainloader, model, optimizer):

    model.train()

    running_loss = 0
    acc = 0

    for batch_idx, (batch) in enumerate(trainloader):

        batch = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3].unsqueeze(1)
            }
        
        optimizer.zero_grad()
        
        output = model.forward(**batch)
        #Cross entropy loss
        loss = output.loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        acc += torch.sum(batch["labels"] == (output.logits.squeeze() >= 0.5))

        break   

    acc = acc / len(trainloader)

    return running_loss, acc

def val_func(valloader, model):

    model.eval()

    running_loss = 0
    acc = 0 

    with torch.no_grad():

        for batch_idx, (batch) in enumerate(valloader):

            batch = {
                    'input_ids': batch[0],
                    'token_type_ids': batch[1],
                    'attention_mask': batch[2],
                    'labels': batch[3].unsqueeze(1)
                }

            output = model.forward(**batch)
            #Cross entropy loss
            loss = output.loss

            running_loss += loss.item()
            acc += torch.sum(batch["labels"] == (output.logits.squeeze() >= 0.5))

            break

        acc = acc / len(valloader)

    return running_loss, acc


def train():

    dm = DisasterDataModule("./data", batch_size=16)
    dm.prepare_data()

    dm.setup()

    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()

    #Import model:
    model = ConvBertForSequenceClassification.from_pretrained(
        'YituTech/conv-bert-base').to(device)

    #EVT: Tjek om den træner hele netværket eller kun det nye classifier layer. 

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    steps = 0
    epoch = int(args.epoch)
    train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst = [], [], [], []

    for e in range(int(args.epoch)):

        loss, acc = train_func(trainloader, model, optimizer)
        train_loss_lst.append(loss)
        train_acc_lst.append(acc)

        loss, acc = val_func(valloader, model)
        val_loss_lst.append(loss)
        val_acc_lst.append(acc)

        to_print = [
          f"{terminal_colors.HEADER}EPOCH %03d"          % e,
          f"{terminal_colors.OKBLUE}TRAIN LOSS=%4.4f"            % train_loss_lst[-1], 
          f"{terminal_colors.OKGREEN}TRAIN ACC=%4.4f"           % train_acc_lst[-1], 
          f"{terminal_colors.OKCYAN}VAL LOSS=%4.4f"            % val_loss_lst[-1], 
          f"{terminal_colors.WARNING}VAL ACC=%4.4f"           % val_acc_lst[-1], 
        ]
        print(" ".join(to_print))

        break

        
    #torch.save(model.state_dict(), 'models/' + str(args.model_version) + '_checkpoint.pth')

    plt.figure(figsize = (8,12))
    plt.plot(range(1,len(train_loss_lst)+1), train_loss_lst, label = 'train loss', color = 'blue')
    plt.plot(range(1,len(val_loss_lst)+1), val_loss_lst, label = 'validation loss', color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss - Disaster tweets convBERT model')
    plt.savefig('reports/figures/train_loss.png')

    plt.figure(figsize = (8,12))
    plt.plot(range(1,len(train_acc_lst)+1), train_acc_lst, label = 'train accuracy', color = 'blue')
    plt.plot(range(1,len(val_acc_lst)+1), val_acc_lst, label = 'validation accuracy', color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy - Disaster tweets convBERT model')
    plt.savefig('reports/figures/train_acc.png')


if __name__ == '__main__':
    train()
