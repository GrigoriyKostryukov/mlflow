import json

import torch
import torch.nn as nn
from torchmetrics import Accuracy

# Load your model definition here
from models.my_model import NeuralNetwork


def create_training_setup_from_json(config_path):
    """Creates a training setup based on a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    device = config["device"]
    epochs = config["epochs"]

    # Check if cuda is available (it might not be on the machine where you run this)
    device = (
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    model = NeuralNetwork(**config["model"]["parameters"]).to(
        device
    )  # Assumes NeuralNetwork accepts parameters
    loss_fn = getattr(
        nn, config["loss_function"]["type"]
    )()  # Dynamically get loss function
    metric_fn = Accuracy(**config["metric_function"]["parameters"]).to(device)
    optimizer = getattr(torch.optim, config["optimizer"]["type"])(
        model.parameters(), **config["optimizer"]["parameters"]
    )

    return model, loss_fn, metric_fn, optimizer, epochs, device


# Example usage:
config_path = "config.json"  # Replace with your config file path
model, loss_fn, metric_fn, optimizer, epochs, device = (
    create_training_setup_from_json(config_path)
)

print(f"Using device: {device}")
print(f"Model: {model}")
print(f"Loss function: {loss_fn}")
print(f"Metric function: {metric_fn}")
print(f"Optimizer: {optimizer}")
print(f"Epochs: {epochs}")
