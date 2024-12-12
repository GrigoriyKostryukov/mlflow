import torch
import torch.nn as nn
from torchmetrics import Accuracy

from models.my_model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
epochs = 3
loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
