import pickle

import torch
import torch.nn as nn

from constants import T_START, T_END
from estimate_util import *
from nn_util import train_FNN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fp: pd.DataFrame = pickle.load(open("../stash/feature_panel.pkl", "rb"))
    fp["Year"] = (fp["Date"] / 10000).astype(int)
    estimated_dict: Dict[str, pd.DataFrame] = pickle.load(
        open("../stash/e_d.pkl", "rb")
    )
    expand_estimated_dict(estimated_dict, "RV_FNN")

    criterion = nn.MSELoss()

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
            model = train_FNN(training_p_v, period)
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
                estimated.loc[estimated["Year"] == t, "RV_FNN"] = outputs.view(
                    -1
                ).numpy()

    pickle.dump(estimated_dict, open("../stash/e_d_1.pkl", "wb"))


if __name__ == "__main__":
    main()
