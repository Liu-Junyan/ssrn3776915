from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from constants import FEATURE_SET_ALL, T_START, T_END, Period
import pickle
import pandas as pd
from estimate_util import *

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


# model = FeedForwardNeuralNetwork().to(device)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

fp: pd.DataFrame = pickle.load(open("../feature_panel.pkl", "rb"))
fp["Year"] = (fp["Date"] / 10000).astype(int)
estimated_dict: Dict[str, pd.DataFrame] = pickle.load(open("../e_d.pkl", "rb"))
expand_estimated_dict(estimated_dict, "RV_FNN")


def train(training_panel: pd.DataFrame, period: Period) -> FeedForwardNeuralNetwork:
    model = FeedForwardNeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    response = f"RV_res^{period.name}"
    epoch = 2000
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
            print(f"epoch {i} / {epoch},\t loss = {loss.item():.4f}")
        if should_early_stop(epoch_MSE_profile, i - 1, 30, 0.0001):
            break
    return model


for t in range(T_START, T_END):
    training_panel = fp[fp["Year"] < t].copy()
    testing_panel = fp[fp["Year"] == t].copy()

    training_mean = training_panel[FEATURE_SET_ALL].mean()
    training_std = training_panel[FEATURE_SET_ALL].std()
    training_panel[FEATURE_SET_ALL] = standardize(
        training_panel[FEATURE_SET_ALL], training_mean, training_std
    )
    testing_panel[FEATURE_SET_ALL] = standardize(
        testing_panel[FEATURE_SET_ALL], training_mean, training_std
    )
    for period in Period:
        print(f"Year {t}, series {period.name}: ")
        response = f"RV_res^{period.name}"
        estimated = estimated_dict[period.name]
        training_p_v = validate_panel(training_panel, response)
        testing_p_v = validate_panel(testing_panel, response)
        model = train(training_p_v, period)
        with torch.no_grad():
            inputs = torch.tensor(
                testing_p_v[FEATURE_SET_ALL].values, dtype=torch.float32
            ).to(device)
            targets = (
                torch.tensor(testing_p_v[response].values, dtype=torch.float32)
                .view(-1, 1)
                .to(device)
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(f"Testing loss: {loss}")
            estimated.loc[estimated["Year"] == t, "RV_FNN"] = outputs.view(-1).numpy()


pickle.dump(estimated_dict, open("../e_d_1.pkl", "wb"))

if __name__ == "__main__":
    pass
