from pathlib import Path

import torch


def test_prepare_data(datamodule):

    assert Path("./data/raw/test.csv").exists()
    assert Path("./data/raw/train.csv").exists()

    assert Path("./data/interim/test_cleaned.csv").exists()
    assert Path("./data/interim/train_cleaned.csv").exists()

    assert Path("./data/processed/test.pt").exists()
    assert Path("./data/processed/train.pt").exists()


def test_setup(datamodule):

    datamodule.setup(stage="fit")

    assert len(datamodule.train_dataset) == 6091
    assert len(datamodule.val_dataset) == 1522
    assert datamodule.train_dataset.indices != datamodule.val_dataset.indices

    full_dataset = datamodule.train_dataset.dataset

    assert len(full_dataset.tensors) == 4
    assert [x.dtype is torch.int64 for x in full_dataset.tensors]

    datamodule.teardown(stage="fit")

    datamodule.setup(stage="test")

    assert len(datamodule.test_dataset) == 3263
    assert len(datamodule.test_dataset.tensors) == 3
    assert [x.dtype is torch.int64 for x in datamodule.test_dataset.tensors]

    datamodule.teardown(stage="test")


def test_dataloaders(datamodule):

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    train_batch = next(iter(train_dataloader))
    assert all(x.shape == (32, 512) for x in train_batch[:3])
    assert train_batch[3].shape == torch.Size([32])

    val_batch = next(iter(val_dataloader))
    assert all(x.shape == (32, 512) for x in val_batch[:3])
    assert val_batch[3].shape == torch.Size([32])

    test_batch = next(iter(test_dataloader))
    assert all(x.shape == (32, 512) for x in test_batch)
