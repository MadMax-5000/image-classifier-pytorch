import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import Net
from src.train import evaluate, train, train_one_epoch, validate


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=20, num_classes=3):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 128, 128)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


class TestTrainingFunctions:
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def model(self, device):
        return Net(num_classes=3).to(device)

    @pytest.fixture
    def loader(self):
        return DataLoader(DummyDataset(size=20), batch_size=4)

    def test_train_one_epoch(self, model, loader, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
        assert loss >= 0

    def test_validate(self, model, loader, device):
        criterion = nn.CrossEntropyLoss()
        loss, acc = validate(model, loader, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100

    def test_evaluate(self, model, loader, device):
        criterion = nn.CrossEntropyLoss()
        loss, acc = evaluate(model, loader, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100

    def test_train_function(self, model, loader, device):
        val_loader = DataLoader(DummyDataset(size=10), batch_size=4)

        history = train(model, loader, val_loader, epochs=2, lr=0.001, device=device)

        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2

    def test_training_reduces_loss(self, model, loader, device):
        val_loader = DataLoader(DummyDataset(size=10), batch_size=4)

        history = train(model, loader, val_loader, epochs=3, lr=0.01, device=device)

        assert history["train_loss"][-1] < history["train_loss"][0]


class TestTrainingWithRealData:
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_training_on_small_dataset(self, device):
        model = Net(num_classes=3).to(device)
        train_loader = DataLoader(DummyDataset(size=16), batch_size=4)
        val_loader = DataLoader(DummyDataset(size=8), batch_size=4)

        history = train(
            model, train_loader, val_loader, epochs=2, lr=0.01, device=device
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

        final_acc = history["train_acc"][-1]
        assert 0 <= final_acc <= 100
