import pytest
import torch
from torch import nn

from src.model import Net


class TestModel:
    def test_model_initialization(self):
        model = Net(num_classes=3)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_output_shape(self):
        model = Net(num_classes=3)
        batch_size = 4
        x = torch.randn(batch_size, 3, 128, 128)
        output = model(x)
        assert output.shape == (batch_size, 3)

    def test_model_forward_pass(self):
        model = Net(num_classes=3)
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        assert not torch.isnan(output).any()

    def test_model_different_num_classes(self):
        for num_classes in [2, 3, 5, 10]:
            model = Net(num_classes=num_classes)
            x = torch.randn(2, 3, 128, 128)
            output = model(x)
            assert output.shape == (2, num_classes)

    def test_model_trainable_parameters(self):
        model = Net(num_classes=3)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_model_dropout(self):
        model = Net(num_classes=3, dropout=0.5)
        model.train()
        x = torch.randn(4, 3, 128, 128)
        out1 = model(x)
        out2 = model(x)
        assert not torch.equal(out1, out2)

        model.eval()
        out_eval = model(x)
        assert torch.equal(out_eval, out_eval)

    def test_model_device_transfer(self):
        model = Net(num_classes=3)
        x = torch.randn(1, 3, 128, 128)

        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            output = model(x)
            assert output.device.type == "cuda"
        else:
            output = model(x)
            assert output.device.type == "cpu"

    def test_model_input_size_variations(self):
        model = Net(num_classes=3)
        for size in [64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape == (1, 3)
