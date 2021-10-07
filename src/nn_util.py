import numpy as np
import pandas as pd
import torch
from torch import nn as nn

from constants import FEATURE_SET_ALL, Period
from estimate_util import should_early_stop

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.MSELoss()


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(len(FEATURE_SET_ALL), 5)  # 19 -> 5
        self.fc2 = nn.Linear(5, 2)  # 5 -> 2
        self.fc3 = nn.Linear(2, 1)  # 2-> 1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_FNN(training_panel: pd.DataFrame, period: Period) -> FeedForwardNeuralNetwork:
    """"Train a feed-forward neural network

    Args:
        training_panel (pd.DataFrame): Validated training panel.
        period (Period): Forecast horizon.

    Returns:
        FeedForwardNeuralNetwork: A trained network.
    """
    model = FeedForwardNeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    response = f"RV_res^{period.name}"
    epoch = 10000
    epoch_MSE_profile = np.ndarray(shape=(epoch,))
    inputs = torch.tensor(
        training_panel[FEATURE_SET_ALL].values, dtype=torch.float32
    ).to(
        device
    )  # shape: n * 19
    targets = (
        torch.tensor(training_panel[response].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )  # shape: n * 1

    for i in range(1, epoch + 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_MSE_profile[i - 1] = loss.item()
        if i % 10 == 0:
            print(f"epoch {i} / {epoch},\t loss = {loss.item():.5f}")
        if should_early_stop(epoch_MSE_profile, i - 1, 50, 0.00001):
            break
    return model
