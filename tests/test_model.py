import torch
from model.network import Net

def test_model_forward():
    model = Net()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
