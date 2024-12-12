import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader

import mlflow


def train(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Trains a PyTorch model and logs metrics to MLflow.

    Args:
      dataloader: DataLoader for the training data.
      model: PyTorch model to train.
      loss_fn: Loss function.
      metrics_fn: Function to calculate metrics (e.g., accuracy).
      optimizer: Optimizer to use.
      device: Device to run the training on ('cpu' or 'cuda').
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred: torch.Tensor = model(X)
        loss: torch.Tensor = loss_fn(pred, y)
        accuracy: torch.Tensor = metrics_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val: float = loss.item()
            accuracy_val: float = accuracy.item()
            mlflow.log_metric("loss", loss_val, step=(batch // 100))
            mlflow.log_metric("accuracy", accuracy_val, step=(batch // 100))
            print(
                f"loss: {loss_val:.3f} accuracy: {accuracy_val:.3f} "
                f"[{batch} / {len(dataloader)}]"
            )
            logging.info(
                f"loss: {loss_val:.3f} accuracy: {accuracy_val:.3f} "
                f"[{batch} / {len(dataloader)}]"
            )
