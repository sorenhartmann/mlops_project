import re
from pathlib import Path

import pandas as pd
import torch
from kaggle.api.kaggle_api_extended import KaggleApi
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from transformers import ConvBertTokenizer


class DisasterDataModule(LightningDataModule):

    train_file = "train.pt"
    test_file = "test.pt"

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DisasterDataModule")
        parser.add_argument("--data_dir", default="./data")
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--val_frac", default=0.2, type=float)
        return parent_parser

    def __init__(self, data_dir: str, batch_size=32, val_frac=0.2, **kwargs):

        super().__init__()
        self.root = Path(data_dir)

        self.batch_size = batch_size
        self.val_frac = val_frac

    def prepare_data(self):

        self.download()
        self.clean_text()
        self.tokenize()

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            train_tensors = torch.load(self.processed_folder / self.train_file)
            train_dataset_full = TensorDataset(*train_tensors)

            # Train/validation split
            val_size = int(len(train_dataset_full) * self.val_frac)
            sizes = [len(train_dataset_full) - val_size, val_size]

            # FIXME: Seed?
            self.train_dataset, self.val_dataset = random_split(
                train_dataset_full, sizes
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:

            test_tensors = torch.load(self.processed_folder / self.test_file)
            self.test_dataset = TensorDataset(*test_tensors)

    @property
    def raw_folder(self) -> Path:
        return self.root / "raw"

    @property
    def interim_folder(self) -> Path:
        return self.root / "interim"

    @property
    def processed_folder(self) -> Path:
        return self.root / "processed"

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def download(self):

        files = ["train.csv", "test.csv"]

        if all((self.raw_folder / file_name).exists() for file_name in files):
            return

        api = KaggleApi()
        api.authenticate()

        for file_name in files:
            api.competition_download_file(
                "nlp-getting-started", file_name, path=self.raw_folder
            )

        assert all(
            (self.raw_folder / file_name).exists() for file_name in files
        )

    def clean_text(self):

        files = ["train_cleaned.csv", "test_cleaned.csv"]
        if all(
            (self.interim_folder / file_name).exists() for file_name in files
        ):
            return

        train = pd.read_csv(self.raw_folder / "train.csv")
        test = pd.read_csv(self.raw_folder / "test.csv")

        # Fill no location and keyword with "no_location", "no_keyword"
        for df in [train, test]:
            for col in ["keyword", "location"]:
                df[col] = df[col].fillna(f"no_{col}")

        # Clean text using data_cleaning.py
        from src.data.substitutions import substitutions

        for a, b in substitutions:
            use_regex = True if type(a) is re.Pattern else False
            train["text"] = train["text"].str.replace(a, b, regex=use_regex)
            test["text"] = test["text"].str.replace(a, b, regex=use_regex)

        # Create data/preprocessed folder if it does not exists
        self.interim_folder.mkdir(exist_ok=True)
        self.processed_folder.mkdir(exist_ok=True)

        # Dump new processed data:
        train.to_csv(self.interim_folder / "train_cleaned.csv")
        test.to_csv(self.interim_folder / "test_cleaned.csv")

    def tokenize(self):

        files = [self.train_file, self.test_file]

        if all(
            (self.processed_folder / file_name).exists() for file_name in files
        ):
            return

        train = pd.read_csv(self.interim_folder / "train_cleaned.csv")
        test = pd.read_csv(self.interim_folder / "test_cleaned.csv")

        tokenizer = ConvBertTokenizer.from_pretrained(
            "YituTech/conv-bert-base"
        )

        train_tokenized = tokenizer(
            train["text"].tolist(), padding="max_length", truncation=True
        )

        test_tokenized = tokenizer(
            test["text"].tolist(), padding="max_length", truncation=True
        )

        train_tensors = (
            torch.tensor(train_tokenized.input_ids),
            torch.tensor(train_tokenized.token_type_ids),
            torch.tensor(train_tokenized.attention_mask),
            torch.tensor(train.target, dtype=torch.long),
        )

        test_tensors = (
            torch.tensor(test_tokenized.input_ids),
            torch.tensor(test_tokenized.token_type_ids),
            torch.tensor(test_tokenized.attention_mask),
        )

        torch.save(train_tensors, self.processed_folder / self.train_file)
        torch.save(test_tensors, self.processed_folder / self.test_file)


if __name__ == "__main__":

    dm = DisasterDataModule("./data", batch_size=32)

    dm.prepare_data()
    dm.setup()

    for batch in tqdm(dm.train_dataloader()):
        pass
    for batch in tqdm(dm.val_dataloader()):
        pass

    dm.teardown()
