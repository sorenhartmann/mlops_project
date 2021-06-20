import pytest

from src.data.datamodule import DisasterDataModule


@pytest.fixture()
def data_module():

    data_module = DisasterDataModule("./data")

    return data_module


@pytest.fixture()
def datasets(data_module: DisasterDataModule):

    data_module.prepare_data()
    data_module.setup()

    return (
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    )


def test_lengths(datasets):

    train_dataset, val_dataset, test_dataset = datasets

    assert len(train_dataset) == 6091
    assert len(val_dataset) == 1522
    assert len(test_dataset) == 3263


def test_train_val(datasets):

    train_dataset, val_dataset, test_dataset = datasets

    assert not train_dataset.indices == val_dataset.indices
