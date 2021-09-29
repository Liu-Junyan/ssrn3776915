from typing import Dict
import numpy as np
import pandas as pd
import itertools
from sklearn import linear_model
from constant import Period, FEATURE_SET
from estimate_util import *
import pickle


def estimate_HAR(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    """Estimate by HAR.

    Args:
        fp (pd.DataFrame): A large panel of features and response variables.
        estimated_dict (Dict[str, pd.DataFrame]): A dict of estimated results. MUTATED.
    """
    lm = linear_model.LinearRegression()
    for period in Period:
        response = f"RV_res^{period.name}"
        predicted = f"RV_HAR^{period.name}"
        estimated = estimated_dict[period.name]
        estimated.insert(len(estimated.columns), predicted, np.nan)

        for t in range(2008, 2017):
            training_panel = fp[fp["Year"] < t]
            testing_panel = fp[fp["Year"] == t]
            training_p_valid = validate_panel(training_panel, response)
            lm.fit(training_p_valid[FEATURE_SET], training_p_valid[response])
            testing_p_valid = validate_panel(testing_panel, response)
            estimated.loc[estimated["Year"] == t, predicted] = lm.predict(
                testing_p_valid[FEATURE_SET]
            )


def main():
    fp: pd.DataFrame = pd.read_pickle("../feature_panel.pkl")
    fp["Year"] = (fp["Date"] / 10000).astype(int)
    estimated_dict: Dict[str, pd.DataFrame] = {
        period.name: fp[["Stock", "Date", "Year", f"RV_res^{period.name}"]].dropna()
        for period in Period
    }
    # First use HAR to get benchmark results.
    # estimate_HAR(fp, estimated_dict)

    # pickle.dump(estimated_dict, open('e_d.pkl', 'wb'))
    estimated_dict = pickle.load(open("../e_d.pkl", "rb"))

    # Under training-validation-testing scheme, t = 2008...2015. (testing: 2007...t-1; validation: t; testing: t+1)
    for t in range(2008, 2016):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        training_mean = training_panel[FEATURE_SET].mean()
        training_std = training_panel[FEATURE_SET].std()

        training_panel[FEATURE_SET] = (
            training_panel[FEATURE_SET] - training_mean
        ) / training_std
        validation_panel[FEATURE_SET] = (
            validation_panel[FEATURE_SET] - training_mean
        ) / training_std
        testing_panel[FEATURE_SET] = (
            testing_panel[FEATURE_SET] - training_mean
        ) / training_std

        for period in Period:
            pass

        # for response in RESPONSE_SET:
        #     augmented = list(itertools.chain(*[FEATURE_SET, [response]]))
        #     training_panel[augmented]

    pass


if __name__ == "__main__":
    main()
