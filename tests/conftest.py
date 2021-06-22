import pytest

from src.data.datamodule import DisasterDataModule
from src.models.model import ConvBert


@pytest.fixture(scope="session")
def datamodule():

    datamodule = DisasterDataModule("./data", batch_size=32)
    datamodule.prepare_data()

    return datamodule


@pytest.fixture(scope="session")
def small_batch():

    datamodule = DisasterDataModule("./data", batch_size=4)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))

    yield batch

    datamodule.teardown(stage="fit")


@pytest.fixture(scope="session")
def model():
    model = ConvBert(lr=0.001, fine_tune_layers=4, last_layer_dropout=0.1)
    return model
