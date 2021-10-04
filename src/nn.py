import torch
import torch.nn as nn
from constants import FEATURE_SET_ALL
import pickle
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


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


model = FeedForwardNeuralNetwork()
model = model.to(device)

fp = pickle.load(open("../feature_panel.pkl", "rb"))

if __name__ == "__main__":
    pass
