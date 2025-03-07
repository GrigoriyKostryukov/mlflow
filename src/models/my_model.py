import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self,
    ) -> None:  # Corrected method name and added type hint
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Added type hints
        x = self.flatten(x)
        logits: torch.Tensor = self.linear_relu_stack(x)  # Added type hint
        return logits
