from data.dataset import get_dataloaders

def test_dataloaders():
    train, test = get_dataloaders()
    assert len(train) > 0
    assert len(test) > 0
