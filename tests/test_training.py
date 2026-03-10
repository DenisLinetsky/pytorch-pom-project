from model.network import Net
from training.trainer import train
from data.dataset import get_dataloaders

def test_training_step():
    train_loader, _ = get_dataloaders(batch_size=16)
    model = Net()
    trained = train(model, train_loader, epochs=1)
    assert trained is not None
