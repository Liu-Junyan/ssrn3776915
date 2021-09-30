from typing import Dict
import numpy as np
import pandas as pd
from sklearn import linear_model
from constants import Period, FEATURE_SET
from estimate_util import *
import pickle


def estimate_HAR(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    """Estimate by HAR.

    Args:
        fp (pd.DataFrame): A panel of features and response variables.
        estimated_dict (Dict[str, pd.DataFrame]): A dict of estimated results. MUTATED.
    """
    lm = linear_model.LinearRegression()
    expand_estimated_dict(estimated_dict, "RV_HAR")
    for period in Period:
        response = f"RV_res^{period.name}"
        predicted = f"RV_HAR^{period.name}"
        estimated = estimated_dict[period.name]

        for t in range(2004, 2022):
            training_panel = fp[fp["Year"] < t]
            testing_panel = fp[fp["Year"] == t]
            training_p_v = validate_panel(training_panel, response)
            testing_p_v = validate_panel(testing_panel, response)
            lm.fit(training_p_v[FEATURE_SET], training_p_v[response])
            estimated.loc[estimated["Year"] == t, predicted] = lm.predict(
                testing_p_v[FEATURE_SET]
            )


def estimate_LASSO(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    lasso = linear_model.Lasso()
    expand_estimated_dict(estimated_dict, "RV_LASSO")
    # Under training-validation-testing scheme, t = 2008...2015. (testing: 2007...t-1; validation: t; testing: t+1)
    for t in range(2008, 2016):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        training_mean = training_panel[FEATURE_SET].mean()
        training_std = training_panel[FEATURE_SET].std()

        training_panel[FEATURE_SET] = standardize(
            training_panel[FEATURE_SET], training_mean, training_std
        )
        validation_panel[FEATURE_SET] = standardize(
            validation_panel[FEATURE_SET], training_mean, training_std
        )
        testing_panel[FEATURE_SET] = standardize(
            testing_panel[FEATURE_SET], training_mean, training_std
        )

        for period in Period:
            response = f"RV_res^{period.name}"
            training_p_v = validate_panel(training_panel, response)
            validation_p_v = validate_panel(validation_panel, response)
            testing_p_v = validate_panel(testing_panel, response)
            predicted = f"RV_LASSO^{period.name}"
            estimated_copy = estimated_dict[period.name].copy()
            estimated_copy = estimated_copy[estimated_copy["Year"] == t]

            lmbda = lasso_grid_search(
                training_p_v, validation_p_v, estimated_copy, period.name
            )
            print(f"Lambda for year {t} series {period.name} is {lmbda}")
            lasso.alpha = lmbda
            estimated = estimated_dict[period.name]
            lasso.fit(training_p_v[FEATURE_SET], training_p_v[response])
            estimated.loc[estimated["Year"] == t + 1, predicted] = lasso.predict(
                testing_p_v[FEATURE_SET]
            )


def main():
    fp: pd.DataFrame = pd.read_pickle("../feature_panel.pkl")
    fp["Year"] = (fp["Date"] / 10000).astype(int)
    estimated_dict: Dict[str, pd.DataFrame] = {
        period.name: fp[["Stock", "Date", "Year", f"RV_res^{period.name}"]].dropna()
        for period in Period
    }
    # First use HAR to get benchmark results.
    estimate_HAR(fp, estimated_dict)

    pickle.dump(estimated_dict, open("../e_d.pkl", "wb"))
    estimated_dict = pickle.load(open("../e_d.pkl", "rb"))

    estimate_LASSO(fp, estimated_dict)
    pickle.dump(estimated_dict, open("../e_d_1.pkl", "wb"))

    pass


if __name__ == "__main__":
    main()
