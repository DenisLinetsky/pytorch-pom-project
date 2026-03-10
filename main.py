from data.dataset import get_dataloaders
from model.network import Net
from training.trainer import train
from evaluation.evaluator import evaluate
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = get_dataloaders()
model = Net()

model = train(model, train_loader, epochs=2, device=device)
accuracy = evaluate(model, test_loader, device=device)

print("Accuracy:", accuracy)
