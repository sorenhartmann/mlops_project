import torch
from transformers import ConvBertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split


def build_train_val_loader(df, batch_size, device, val_size):

    tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')

    x = tokenizer(df["text"].to_numpy().tolist(),
                  padding="max_length",
                  truncation=True)

    ds = TensorDataset(
        torch.tensor(x.input_ids).to(device),
        torch.tensor(x.token_type_ids).to(device),
        torch.tensor(x.attention_mask).to(device),
        torch.tensor(df.target).float().to(device))

    # split
    val_ds_size = int(len(ds) * val_size)
    sizes = [len(ds) - val_ds_size, val_ds_size]
    train_ds, val_ds = random_split(ds, sizes)

    trainloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(val_ds, shuffle=True, batch_size=batch_size)

    return trainloader, valloader


def build_test_loader(df, batch_size, device):

    tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')

    x = tokenizer(df["text"].to_numpy().tolist(),
                  padding="max_length",
                  truncation=True)

    ds = TensorDataset(
        torch.tensor(x.input_ids).to(device),
        torch.tensor(x.token_type_ids).to(device),
        torch.tensor(x.attention_mask).to(device))

    testloader = DataLoader(ds, shuffle=True, batch_size=batch_size)

    return testloader